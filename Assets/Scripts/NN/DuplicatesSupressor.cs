using NN;
using System;
using System.Collections.Generic;

public static class DuplicatesSupressor
{
    const float OVERLAP_TRESHOLD = 0.2f;

    static public void RemoveDuplicats(List<ResultBox> boxes)
    {
        if (boxes.Count == 0)
            return;

        int classesNum = boxes[0].classes.Length;
        for (int classIndex = 0; classIndex < classesNum; classIndex++)
        {
            RemoveDuplicatesForClass(boxes, classIndex);
        }
    }

    private static void RemoveDuplicatesForClass(List<ResultBox> boxes, int classIndex)
    {
        List<ResultBox> sortedBoxes = SortBoxesByClassValue(boxes, classIndex);

        for (int i = 0; i < boxes.Count; i++)
        {
            ResultBox i_box = sortedBoxes[i];
            if (i_box.classes[classIndex] == 0)
                continue;

            for (int j = i + 1; j < boxes.Count; j++)
            {
                ResultBox j_box = sortedBoxes[j];
                if (NNUtils.BoxesIOU(i_box.rect, j_box.rect) >= OVERLAP_TRESHOLD)
                    j_box.classes[classIndex] = 0;
            }
        }
    }

    private static List<ResultBox> SortBoxesByClassValue(List<ResultBox> boxes, int classIndex)
    {
        Comparison<ResultBox> boxClassValueComparer =
            (box1, box2) => box2.classes[classIndex].CompareTo(box1.classes[classIndex]);
        boxes.Sort(boxClassValueComparer);
        return boxes;
    }
}