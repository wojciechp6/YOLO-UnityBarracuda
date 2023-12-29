using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class NNUtils
{
    static float _interval_overlap(float box1_min, float box1_max, float box2_min, float box2_max)
    {
        if (box2_min < box1_min)
        {
            if (box2_max < box1_min)
                return 0;
            else
                return Mathf.Min(box1_max, box2_max) - box1_min;
        }
        else
        {
            if (box1_max < box2_min)
                return 0;
            else
                return Mathf.Min(box1_max, box2_max) - box2_min;
        }
    }

    public static float BoxesIOU(Rect box1, Rect box2)
    {
        float intersect_w = _interval_overlap(box1.x, box1.x + box1.width, box2.x, box2.x + box2.width);
        float intersect_h = _interval_overlap(box1.y, box1.y + box1.height, box2.y, box2.y + box2.height);

        float intersect = intersect_w * intersect_h;

        float union = box1.width * box1.height + box2.width * box2.height - intersect;
        return intersect / union;
    }

}
