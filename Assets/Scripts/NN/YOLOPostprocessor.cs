using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

namespace NN
{
     static public class YOLOPostprocessor
    {
        const float DISCARD_TRESHOLD = 0.1f;
        const int classesNum = 20;
        const int BoxesPerCell = 5;
        const int inputWidthHeight = 416;
        static readonly float[] anchors = new[] { 1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f };

        static IOps cpuOps;

        static YOLOPostprocessor()
        {
            cpuOps = BarracudaUtils.CreateOps(WorkerFactory.Type.CSharp);
        }

        static public List<ResultBox> DecodeNNOut(Tensor output)
        {
            List<ResultBox> boxes = new();
            float[,,,] array = ReadOutputToArray(output);
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

        private static float[,,,] ReadOutputToArray(Tensor output)
        {
            var reshapedOutput = output.Reshape(new[] { output.height, output.width, BoxesPerCell, 25 });
            var array = TensorToArray4D(reshapedOutput);
            reshapedOutput.Dispose();
            return array;
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
            
            const int boxCenterXIndex = 0;
            const int boxCenterYIndex = 1;
            const int boxWidthIndex = 2;
            const int boxHeightIndex = 3;

            float boxCenterX = (x_cell + Sigmoid(data[y_cell, x_cell, box, boxCenterXIndex])) * normalizeRatio;
            float boxCenterY = (y_cell + Sigmoid(data[y_cell, x_cell, box, boxCenterYIndex])) * normalizeRatio;
            float boxWidth = Mathf.Exp(data[y_cell, x_cell, box, boxWidthIndex]) * anchors[2 * box] * normalizeRatio;
            float boxHeight = Mathf.Exp(data[y_cell, x_cell, box, boxHeightIndex]) * anchors[2 * box + 1] * normalizeRatio;

            float box_x = boxCenterX - boxWidth / 2;
            float box_y = boxCenterY - boxHeight / 2;
            return new Rect(box_x, box_y, boxWidth, boxHeight);
        }

        static private float Sigmoid(float value)
        {
            return 1f / (1f + Mathf.Exp(-value));
        }

        static private float[] Softmax(float[] values)
        {
            Tensor inputTensor = new(1, values.Length, values);
            float[] output = cpuOps.Softmax(inputTensor, axis: -1).AsFloats();
            inputTensor.Dispose();
            return output;
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