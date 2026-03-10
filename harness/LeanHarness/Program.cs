using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using QuantConnect;
using QuantConnect.Configuration;
using QuantConnect.Lean.Engine;
using QuantConnect.Lean.Engine.DataFeeds;
using QuantConnect.Logging;
using QuantConnect.Util;

namespace LeanHarness
{
    internal class HarnessWorkerThread : WorkerThread { }

    /// <summary>
    /// No-op log handler that discards all LEAN log messages for maximum performance.
    /// Eliminates handler-side string formatting, IO writes, and Python-side stderr drain load.
    /// </summary>
    internal class NullLogHandler : ILogHandler
    {
        public void Trace(string text) { }
        public void Debug(string text) { }
        public void Error(string text) { }
        public void Dispose() { }
    }

    public class Program
    {
        private static TextWriter _jsonOut;

        static void Main(string[] args)
        {
            // Save real stdout for JSON protocol, redirect Console.Out to stderr
            // so all LEAN logging goes to stderr and doesn't corrupt JSON output.
            _jsonOut = Console.Out;
            Console.SetOut(Console.Error);

            Thread.CurrentThread.Name = "Harness Main Thread";

            // Determine the LEAN launcher directory for Composer assembly scanning.
            // Default: /Lean/Launcher/bin/Debug (the standard location inside the container).
            // Can be overridden via LEAN_LAUNCHER_DIR env var.
            var leanLauncherDir = Environment.GetEnvironmentVariable("LEAN_LAUNCHER_DIR")
                ?? "/Lean/Launcher/bin/Debug";
            Config.Set("composer-dll-directory", leanLauncherDir);

            // One-time initialization
            OS.Initialize();

            Console.Error.WriteLine($"[harness] Ready — composer-dll-directory={leanLauncherDir}");
            Console.Error.Flush();

            string line;
            while ((line = Console.In.ReadLine()) != null)
            {
                line = line.Trim();
                if (string.IsNullOrEmpty(line))
                    continue;

                JObject request;
                try
                {
                    request = JObject.Parse(line);
                }
                catch (Exception ex)
                {
                    WriteError("unknown", $"Invalid JSON: {ex.Message}");
                    continue;
                }

                var id = request.Value<string>("id") ?? "unknown";
                var configPath = request.Value<string>("config_path");

                if (string.IsNullOrEmpty(configPath))
                {
                    WriteError(id, "Missing config_path");
                    continue;
                }

                try
                {
                    RunBacktest(id, configPath);
                    WriteOk(id);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[harness] [{id.Substring(0, Math.Min(8, id.Length))}] Exception: {ex.Message}");
                    WriteError(id, ex.Message);
                }
            }

            OS.Dispose();
            Console.Error.WriteLine("[harness] stdin closed — exiting");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void RunBacktest(string id, string configPath)
        {
            var shortId = id.Substring(0, Math.Min(8, id.Length));
            Console.Error.WriteLine($"[harness] [{shortId}] Starting backtest");

            // Determine logging mode. Must be set FIRST — before any Config/Composer
            // calls that trigger Log.Trace() via Config.Get() missing-key warnings.
            var debug = Environment.GetEnvironmentVariable("HARNESS_DEBUG") == "1";
            ILogHandler handler = debug ? (ILogHandler)new ConsoleErrorLogHandler() : new NullLogHandler();
            Log.LogHandler = handler;
            Log.DebuggingEnabled = debug;

            // Reset all state — mirrors LEAN's AlgorithmRunner.RunLocalBacktest()
            Config.Reset();
            // Re-set handler immediately — Config.Reset() may reset Log.LogHandler
            Log.LogHandler = handler;
            Log.DebuggingEnabled = debug;

            Config.MergeCommandLineArgumentsWithConfiguration(
                new Dictionary<string, object> { { "config", configPath } }
            );
            Composer.Instance.Reset();
            SymbolCache.Clear();
            TextSubscriptionDataSourceReader.ClearCache();
            Globals.Reset();

            // Tell Initializer.Start() to use ConsoleErrorLogHandler (no-ops Trace/Debug)
            // instead of its default CompositeLogHandler, so Config.Get() calls inside
            // Initializer.Start() are also suppressed.
            if (!debug)
            {
                Config.Set("log-handler", "ConsoleErrorLogHandler");
                Config.Set("debug-mode", "false");
            }

            LeanEngineSystemHandlers systemHandlers = null;
            LeanEngineAlgorithmHandlers algorithmHandlers = null;

            try
            {
                Initializer.Start();

                // 3) After Initializer.Start(), swap to NullLogHandler for zero overhead
                //    during the actual backtest engine run.
                Log.LogHandler = handler;
                Log.DebuggingEnabled = debug;
                systemHandlers = Initializer.GetSystemHandlers();

                var job = systemHandlers.JobQueue.NextJob(out var assemblyPath);
                if (job == null)
                    throw new Exception("JobQueue returned null job");

                algorithmHandlers = Initializer.GetAlgorithmHandlers();

                var algorithmManager = new AlgorithmManager(false, job);
                systemHandlers.LeanManager.Initialize(
                    systemHandlers, algorithmHandlers, job, algorithmManager
                );

                var engine = new QuantConnect.Lean.Engine.Engine(
                    systemHandlers, algorithmHandlers, false
                );

                // Create a fresh WorkerThread per run — Engine.Run() disposes it
                using var workerThread = new HarnessWorkerThread();
                engine.Run(job, algorithmManager, assemblyPath, workerThread);

                systemHandlers.JobQueue.AcknowledgeJob(job);

                Console.Error.WriteLine($"[harness] [{shortId}] Backtest complete");
            }
            finally
            {
                systemHandlers?.DisposeSafely();
                algorithmHandlers?.DisposeSafely();
                Log.LogHandler.DisposeSafely();
            }
        }

        private static void WriteOk(string id)
        {
            var response = new JObject
            {
                ["id"] = id,
                ["status"] = "ok"
            };
            _jsonOut.WriteLine(response.ToString(Formatting.None));
            _jsonOut.Flush();
        }

        private static void WriteError(string id, string message)
        {
            var response = new JObject
            {
                ["id"] = id,
                ["status"] = "error",
                ["message"] = message
            };
            _jsonOut.WriteLine(response.ToString(Formatting.None));
            _jsonOut.Flush();
        }
    }
}
