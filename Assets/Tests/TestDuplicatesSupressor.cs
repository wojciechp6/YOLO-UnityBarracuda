using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NN;
using NUnit.Framework;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;
using UnityEngine.SocialPlatforms.Impl;
using UnityEngine.TestTools;

public class TestDuplicatesSupressor
{
    [Test]
    public void ReadsTensorSuccessfully()
    {
        List<ResultBox> boxes = new();

        int best = 1;
        float class_score = 0.9f;
        float[] classes = new float[20];
        classes[best] = class_score;
        Rect rect = new(0, 0, 1, 1);

        ResultBox box1 = new ResultBox {
            bestClassIdx = best,
            classes = classes,
            rect = rect
        };
        boxes.Add(box1);

        
        ResultBox box2 = new ResultBox
        {
            bestClassIdx = best,
            classes = classes,
            rect = rect
        };

        DuplicatesSupressor.RemoveDuplicats(boxes);

        Assert.AreEqual(1, boxes.Count);
    }
}
