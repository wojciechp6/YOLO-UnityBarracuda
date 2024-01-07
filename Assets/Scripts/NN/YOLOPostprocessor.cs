using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Profiling;
using static UnityEngine.Networking.UnityWebRequest;
using static UnityEngine.Analytics.IAnalytic;
using System.Linq;

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
            List<ResultBox> boxes = new();
            var reshapedOutput = output.Reshape(new[] { 13, 13, 5, 25 });
            float[] vals = reshapedOutput.AsFloats();
            output.Dispose();
            var array = TensorToArray4D(reshapedOutput);
            int widht = array.GetLength(0);
            int height = array.GetLength(1);

            for (int y_cell = 0; y_cell < height; y_cell++)
            {
                for (int x_cell = 0; x_cell < widht; x_cell++)
                {
                    var cell_boxes = DecodeCell(array, x_cell, y_cell);
                    boxes.AddRange(cell_boxes);
                }
            }

            return boxes;
        }

        private static IEnumerable<ResultBox> DecodeCell(float[,,,] array, int x_cell, int y_cell)
        {
            int boxes = array.GetLength(2); 
            for (int box_index = 0; box_index < boxes; box_index++)
            {
                var box = DecodeBox(array, x_cell, y_cell, box_index);
                if (box.HasValue)
                    yield return box.Value;
            }
        }

        static private ResultBox? DecodeBox(float[,,,] array, int x_cell, int y_cell, int box)
        {
            float box_score = DecodeBoxScore(array, x_cell, y_cell, box);
            if (box_score < DISCARD_TRESHOLD)
                return null;

            Rect box_rect = DecodeBoxRectangle(array, x_cell, y_cell, box);
            float[] box_classes = DecodeBoxClasses(array, x_cell, y_cell, box, box_score);
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

        static private float DecodeBoxScore(float[,,,] array, int x_cell, int y_cell, int box)
        {
            const int boxScoreIndex = 4;
            return Sigmoid(array[y_cell, x_cell, box, boxScoreIndex]);
        }

        static private float[] DecodeBoxClasses(float[,,,] array, int x_cell, int y_cell, int box, float box_score)
        {
            float[] box_classes = new float[classesNum];
            const int classesOffset = 5;
            
            for (int i = 0; i < classesNum; i++)
                box_classes[i] = array[y_cell, x_cell, box, i + classesOffset];

            box_classes = Softmax(box_classes);
            box_classes.Update(x => x * box_score);
            return box_classes;
        }

        static private Rect DecodeBoxRectangle(float[,,,] data, int x_cell, int y_cell, int box)
        {
            const float downscaleRatio = 32;
            const float normalizeRatio = downscaleRatio / inputWidthHeight;
            float box_x = (x_cell + Sigmoid(data[y_cell, x_cell, box, 0])) * normalizeRatio;
            float box_y = (y_cell + Sigmoid(data[y_cell, x_cell, box, 1])) * normalizeRatio;
            float box_width = Mathf.Exp(data[y_cell, x_cell, box, 2]) * anchors[2 * box] * normalizeRatio;
            float box_height = Mathf.Exp(data[y_cell, x_cell, box, 3]) * anchors[2 * box + 1] * normalizeRatio;

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
                float[] classValues = boxes.Select((box, i) => box.classes[c]).ToArray();
                int[] sortedIndexes = SortBoxesByClassValue(classValues);

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

        private static int[] SortBoxesByClassValue(float[] values)
        {
            List<KeyValuePair<int, float>> dic = new();
            values.ForEach((x, i) => dic.Add(new KeyValuePair<int, float>(i, x)));
            dic.Sort((pair1, pair2) => pair2.Value.CompareTo(pair1.Value));
            return dic.Select((x, i) => x.Key).ToArray();
        }

        private static float[,,,] TensorToArray4D(this Tensor tensor)
        {
            float[,,,] output = new float[tensor.batch, tensor.height, tensor.width, tensor.channels];
            var data = tensor.AsFloats();
            int bytes = Buffer.ByteLength(data);
            Buffer.BlockCopy(data, 0, output, 0, bytes);
            return output;
        }
    }
}