using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Profiling;

namespace NN
{
    public struct ResultBox
    {
        public Rect rect;
        public float confidence;
        public float[] classes;
        public int bestClassIdx;
    }

    public class YOLOHandler : IDisposable
    {
        NNHandler nn;
        IOps ops;

        Tensor premulTensor;

        PerformanceCounter.StopwatchCounter stopwatch = new PerformanceCounter.StopwatchCounter("Net inference time");

        public YOLOHandler(NNHandler nn)
        {
            this.nn = nn;
            ops = BarracudaUtils.CreateOps(WorkerFactory.Type.Compute);

            var inputWidthHeight = nn.model.inputs[0].shape[5];
            premulTensor = new Tensor(1, 1, new float[] { 255 });

            PerformanceCounter.GetInstance()?.AddCounter(stopwatch);
        }

        public List<ResultBox> Run(Texture2D tex)
        {
            Profiler.BeginSample("YOLO.Run");

            Tensor input = new Tensor(tex);
            var preprocessed = Preprocess(input);
            input.Dispose();

            Tensor output = Execute(preprocessed);
            preprocessed.Dispose();

            var results = Postprocess(output);
            output.Dispose();

            Profiler.EndSample();
            return results;
        }

        public void Dispose()
        {
            premulTensor.Dispose();
        }

        private Tensor Execute(Tensor preprocessed)
        {
            Profiler.BeginSample("YOLO.Execute");

            nn.worker.Execute(preprocessed);
            nn.worker.FlushSchedule();
            var output = nn.worker.PeekOutput();

            Profiler.EndSample();
            return output;
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
            var results = YOLOPostprocessor.DecodeNNOut(x);
            YOLOPostprocessor.RemoveDuplicats(results);
            Profiler.EndSample();
            return results;
        }
    }
}