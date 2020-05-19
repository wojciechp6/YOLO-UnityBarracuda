using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using UnityEngine.Profiling;
using Unity.Jobs;
using Unity.Collections;

public class YOLOHandler : IDisposable
{
    public struct ResultBox
    {
        public Rect rect;
        public float confidence;
        public float[] classes;
        public int bestClassIdx;
    }

    const float DISCARD_TRESHOLD = 0.1f;
    const float OVERLAP_TRESHOLD = 0.2f;

    const int classesNum = 20;
    float[] anchors = new float[] { 1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f };
    const int BoxesPerCell = 5;
    int inputWidthHeight;


    NNHandler nn;
    IOps ops;
    IOps cpuOps;

    Tensor premulTensor;

    PerformanceCounter.StopwatchCounter stopwatch = new PerformanceCounter.StopwatchCounter("Net inference time");

    public YOLOHandler(NNHandler nn)
    {
        this.nn = nn;
        ops = BarracudaUtils.CreateOps(WorkerFactory.Type.Compute);
        cpuOps = BarracudaUtils.CreateOps(WorkerFactory.Type.CSharp);

        inputWidthHeight = nn.model.inputs[0].shape[1];
        premulTensor = new Tensor(1, 1, new float[] { 255 });

        PerformanceCounter.GetInstance()?.AddCounter(stopwatch);
    }

    public List<ResultBox> Run(Texture2D tex)
    {
        Profiler.BeginSample("YOLO.Run");

        Tensor input = new Tensor(tex);

        var preprocessed = Preprocess(input);
        input.Dispose();

        stopwatch.Start();
        Tensor output = Execute(preprocessed);
        stopwatch.Stop();
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
        nn.worker.WaitForCompletion();
        var output = nn.worker.PeekOutput();
        
        Profiler.EndSample();
        return output;
    }

    Tensor Preprocess(Tensor x)
    {
        Profiler.BeginSample("YOLO.Preprocess");
        var preprocessed = ops.Mul(new Tensor[] { x, premulTensor });
        Profiler.EndSample();
        return preprocessed;
    }

    List<ResultBox> Postprocess(Tensor x)
    {
        Profiler.BeginSample("YOLO.Postprocess");
        var results = DecodeNNOut(x);
        RemoveDuplicats(results, OVERLAP_TRESHOLD);
        Profiler.EndSample();
        return results;
    }

    private List<ResultBox> DecodeNNOut(Tensor output)
    {
        float[] data = output.AsFloats();

        int cellSize = output.channels;
        int boxSize = cellSize / BoxesPerCell;
        int boxesNum = output.width * output.height * BoxesPerCell;
        int outputWidth = output.width;

        List<ResultBox> results = new List<ResultBox>();

        for (int y_cell = 0; y_cell < output.height; y_cell++)
        {
            for (int x_cell = 0; x_cell < output.width; x_cell++)
            {
                for (int box = 0; box < BoxesPerCell; box++)
                {
                    int idx = (x_cell + y_cell * output.width) * cellSize + box * boxSize;

                    var result = DecodeBox(data, idx, x_cell, y_cell, box);
                    if (result.HasValue)
                        results.Add(result.Value);
                }
            }
        }

        return results;
    }

    private ResultBox? DecodeBox(float[] data, int startIndex, int x_cell, int y_cell, int box)
    {
        float box_score = Sigmoid(data[startIndex + 4]);
        if (box_score < DISCARD_TRESHOLD)
            return null;

        Rect box_rect = DecodeBoxRectangle(data, startIndex, x_cell, y_cell, box);
        float[] box_classes = DecodeBoxClasses(data, startIndex, box_score);

        int bestClassIdx = box_classes.MaxIdx();

        var result = new ResultBox
        {
            rect = box_rect,
            confidence = box_score,
            bestClassIdx = bestClassIdx,
            classes = box_classes
        };
        return result;
    }

    private float[] DecodeBoxClasses(float[] data, int startIndex, float box_score)
    {
        float[] box_classes = data.GetRange(startIndex + 5, startIndex + 5 + classesNum);
        box_classes = Softmax(box_classes);
        box_classes.Update(x => x * box_score);
        return box_classes;
    }

    private Rect DecodeBoxRectangle(float[] data, int startIndex, int x_cell, int y_cell, int box)
    {
        float box_x = (x_cell + Sigmoid(data[startIndex])) * 32 / inputWidthHeight;
        float box_y = (y_cell + Sigmoid(data[startIndex + 1])) * 32 / inputWidthHeight;
        float box_width = Mathf.Exp(data[startIndex + 2]) * anchors[2 * box] * 32 / inputWidthHeight;
        float box_height = Mathf.Exp(data[startIndex + 3]) * anchors[2 * box + 1] * 32 / inputWidthHeight;

        return new Rect(box_x - box_width / 2, 
            box_y - box_height / 2, box_width, box_height);
    }

    private static float Sigmoid(float value)
    {
        return 1f / (1f + Mathf.Exp(-value));
    }

    private float[] Softmax(float[] values)
    {
        Tensor t = new Tensor(1, values.Length, values);
        var ret = cpuOps.Softmax(t).AsFloats();
        t.Dispose();
        return ret;
    }

    private void RemoveDuplicats(List<ResultBox> boxes, float nms_thresh)
    {
        if (boxes.Count == 0)
            return;

        for (int c = 0; c < classesNum; c++)
        {
            float[] classValues = new float[boxes.Count];
            classValues.Update((x, i) => boxes[i].classes[c]);

            int[] sortedIndexes = _sortIdx(classValues);

            for (int i = 0; i < boxes.Count; i++)
            {
                int i_index = sortedIndexes[i];
                if (boxes[i_index].classes[c] == 0)
                    continue;

                for (int j = i + 1; j < boxes.Count; j++)
                {
                    int j_index = sortedIndexes[j];
                    if (NNUtils.BoxesIOU(boxes[i_index].rect, boxes[j_index].rect) >= nms_thresh)
                        boxes[j_index].classes[c] = 0;
                }
            }
        }
    }

    private static int[] _sortIdx(float[] values)
    {
        List<KeyValuePair<int, float>> dic = new List<KeyValuePair<int, float>>();
        values.ForEach((x, i) => dic.Add(new KeyValuePair<int, float>(i, x)));
        dic.Sort((pair1, pair2) => pair2.Value.CompareTo(pair1.Value));
        return (int[])new int[values.Length].Update((x, i) => dic[i].Key);
    }
}
