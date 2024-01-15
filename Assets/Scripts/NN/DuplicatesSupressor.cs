using NN;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Profiling;

public static class DuplicatesSupressor
{
    const float OVERLAP_TRESHOLD = 0.3f;

    static public List<ResultBox> RemoveDuplicats(List<ResultBox> boxes)
    {
        Profiler.BeginSample("DuplicatesSupressor.RemoveDuplicats");

        if (boxes.Count == 0)
            return boxes;

        List<ResultBox> result_boxes = new();

        for (int classIndex = 0; classIndex < 80; classIndex++)
        {
            var classBoxes = boxes.Where(box => box.bestClassIndex == classIndex).ToList();
            RemoveDuplicatesForClass(boxes);
            classBoxes = classBoxes.Where(box => box.score > 0).ToList();
            result_boxes.AddRange(classBoxes);
        }

        Profiler.EndSample();

        return result_boxes;
    }

    private static void RemoveDuplicatesForClass(List<ResultBox> boxes)
    {
        SortBoxesByScore(boxes);
        for (int i = 0; i < boxes.Count; i++)
        {
            ResultBox i_box = boxes[i];
            if (i_box.score == 0)
                continue;

            for (int j = i + 1; j < boxes.Count; j++)
            {
                ResultBox j_box = boxes[j];
                float iou = IntersectionOverUnion.CalculateIOU(i_box.rect, j_box.rect);
                if (iou >= OVERLAP_TRESHOLD && i_box.score > j_box.score)
                {
                    j_box.score = 0;
                }
            }
        }
    }

    private static List<ResultBox> SortBoxesByScore(List<ResultBox> boxes)
    {
        Comparison<ResultBox> boxClassValueComparer =
            (box1, box2) => box2.score.CompareTo(box1.score);
        boxes.Sort(boxClassValueComparer);
        return boxes;
    }
}