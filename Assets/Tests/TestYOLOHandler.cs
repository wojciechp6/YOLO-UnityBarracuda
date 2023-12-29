using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using NN;
using Unity.Barracuda;
using UnityEditor;
using System.Linq;


public class TestYOLOHandler
{
    const string MODEL_PATH = "Assets/YOLOv2 Tiny.onnx";
    const string IMAGE_PATH = "Assets/Tests/test_image.jpg";
    const float min_confidence = 0.3f;
    private YOLOHandler yolo;

    [SetUp]
    public void Setup()
    {
        NNModel model = AssetDatabase.LoadAssetAtPath<NNModel>(MODEL_PATH);
        NNHandler nnHandler = new(model);
        yolo = new YOLOHandler(nnHandler);
    }

    [TearDown]
    public void TearDown()
    {
        yolo.Dispose();
    }

    [Test]
    public void CreatesSuccessfully()
    {
        Assert.NotNull(yolo);
    }

    [Test]
    public void NotZeroResult()
    {
        // given
        Texture2D image = AssetDatabase.LoadAssetAtPath<Texture2D>(IMAGE_PATH);
        var results = yolo.Run(image);
        Assert.NotZero(results.Count);
    }

    [Test]
    public void TwoConfidentResults()
    {
        // given
        Texture2D image = AssetDatabase.LoadAssetAtPath<Texture2D>(IMAGE_PATH);
        var results = yolo.Run(image);
        var confident_results = results.Where(box => box.classes[box.bestClassIdx] > min_confidence).ToList();
        Assert.AreEqual(2, confident_results.ToList().Count);
    }

    [Test]
    public void ConfidentResultsHasRightClasses()
    {
        // given
        Texture2D image = AssetDatabase.LoadAssetAtPath<Texture2D>(IMAGE_PATH);
        var results = yolo.Run(image);
        var confident_results = results.Where(box => box.classes[box.bestClassIdx] > min_confidence).ToList();
        Assert.AreEqual(19, confident_results[0].bestClassIdx);
        Assert.AreEqual(14, confident_results[1].bestClassIdx);
    }

}
