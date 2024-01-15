using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOHandler : IDisposable
    {
        NNHandler nn;
        public IOps ops;
        Tensor premulTensor;
        PerformanceCounter.StopwatchCounter stopwatch = new PerformanceCounter.StopwatchCounter("Net inference time");

        public YOLOHandler(NNHandler nn)
        {
            this.nn = nn;
            ops = BarracudaUtils.CreateOps(WorkerFactory.Type.ComputePrecompiled);
            premulTensor = new Tensor(1, 1, new float[] { 255 });
            PerformanceCounter.GetInstance()?.AddCounter(stopwatch);
        }

        public List<ResultBox> Run(Texture2D tex)
        {
            Profiler.BeginSample("YOLO.Run");

            Tensor input = new Tensor(tex);
            var preprocessed = Preprocess(input);
            input.tensorOnDevice.Dispose();
            ExecuteNetwork(preprocessed);
            preprocessed.tensorOnDevice.Dispose();
            Tensor output = GetNetwokOutput();
            var results = Postprocess(output);

            Profiler.EndSample();
            return results;
        }

        public void Dispose()
        {
            premulTensor.tensorOnDevice.Dispose();
        }

        private void ExecuteNetwork(Tensor preprocessed)
        {
            Profiler.BeginSample("YOLO.Execute");
            nn.worker.Execute(preprocessed);
            nn.worker.FlushSchedule(blocking: true);
            Profiler.EndSample();
        }

        private Tensor GetNetwokOutput()
        {
            return nn.worker.PeekOutput();
        }

        private Tensor Preprocess(Tensor x)
        {
            Profiler.BeginSample("YOLO.Preprocess");
            var preprocessed = ops.Mul(new[]{ x, premulTensor });
            Profiler.EndSample();
            return preprocessed;
        }

        List<ResultBox> Postprocess(Tensor x)
        {
            Profiler.BeginSample("YOLO.Postprocess");
            var results = YOLOv2Postprocessor.DecodeNNOut(x);
            results = DuplicatesSupressor.RemoveDuplicats(results);
            Profiler.EndSample();
            return results;
        }
    }
}