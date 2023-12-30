using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Profiling;

namespace NN
{
     static public class YOLOPostprocessor
    {
        const float DISCARD_TRESHOLD = 0.1f;
        const float OVERLAP_TRESHOLD = 0.2f;

        const int classesNum = 20;
        static readonly float[] anchors = new[] { 1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f };
        const int BoxesPerCell = 5;
        const int inputWidthHeight = 416;

        static IOps cpuOps;

        static YOLOPostprocessor()
        {
            cpuOps = BarracudaUtils.CreateOps(WorkerFactory.Type.CSharp);
        }

        static public List<ResultBox> DecodeNNOut(Tensor output)
        {
            float[] data = output.AsFloats();

            int cellSize = output.channels;
            int boxSize = cellSize / BoxesPerCell;

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


        static private ResultBox? DecodeBox(float[] data, int startIndex, int x_cell, int y_cell, int box)
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

        static private float[] DecodeBoxClasses(float[] data, int startIndex, float box_score)
        {
            float[] box_classes = data.GetRange(startIndex + 5, startIndex + 5 + classesNum);
            box_classes = Softmax(box_classes);
            box_classes.Update(x => x * box_score);
            return box_classes;
        }

        static private Rect DecodeBoxRectangle(float[] data, int startIndex, int x_cell, int y_cell, int box)
        {
            float box_x = (x_cell + Sigmoid(data[startIndex])) * 32 / inputWidthHeight;
            float box_y = (y_cell + Sigmoid(data[startIndex + 1])) * 32 / inputWidthHeight;
            float box_width = Mathf.Exp(data[startIndex + 2]) * anchors[2 * box] * 32 / inputWidthHeight;
            float box_height = Mathf.Exp(data[startIndex + 3]) * anchors[2 * box + 1] * 32 / inputWidthHeight;

            return new Rect(box_x - box_width / 2,
                box_y - box_height / 2, box_width, box_height);
        }

        static private float Sigmoid(float value)
        {
            return 1f / (1f + Mathf.Exp(-value));
        }

        static private float[] Softmax(float[] values)
        {
            Tensor t = new Tensor(1, values.Length, values);
            var ret = cpuOps.Softmax(t, axis: -1).AsFloats();
            t.Dispose();
            return ret;
        }

        static public void RemoveDuplicats(List<ResultBox> boxes)
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
                        if (NNUtils.BoxesIOU(boxes[i_index].rect, boxes[j_index].rect) >= OVERLAP_TRESHOLD)
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
}