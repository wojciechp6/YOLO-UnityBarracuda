using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using NN;
using Unity.Barracuda;
using UnityEditor;
using System.Linq;
using static UnityEngine.Networking.UnityWebRequest;


public class TestYOLOHandler
{
    const string MODEL_PATH = "Assets/YOLOv2 Tiny.onnx";
    const string IMAGE_PATH = "Assets/Tests/test_image.jpg";
    const float min_confidence = 0.3f;
    private YOLOHandler yolo;
    private Texture2D test_image;

    [SetUp]
    public void Setup()
    {
        NNModel model = AssetDatabase.LoadAssetAtPath<NNModel>(MODEL_PATH);
        NNHandler nnHandler = new(model);
        yolo = new YOLOHandler(nnHandler);
        test_image = AssetDatabase.LoadAssetAtPath<Texture2D>(IMAGE_PATH);

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
        // when
        var results = yolo.Run(test_image);
        Assert.NotZero(results.Count);
    }

    [Test]
    public void TwoConfidentResults()
    {
        // when
        var results = yolo.Run(test_image);
        List<YOLOHandler.ResultBox> confident_results = GetConfidentResults(results);

        // then
        Assert.AreEqual(2, confident_results.Count);
    }

    [Test]
    public void ConfidentResultsHasRightClasses()
    {
        // given
        var results = yolo.Run(test_image);
        var confident_results = GetConfidentResults(results);

        // then
        Assert.AreEqual(19, confident_results[0].bestClassIdx);
        Assert.AreEqual(14, confident_results[1].bestClassIdx); 
    }

    private List<YOLOHandler.ResultBox> GetConfidentResults(List<YOLOHandler.ResultBox> rawResults)
    {
        return rawResults.Where(box => box.classes[box.bestClassIdx] > min_confidence).ToList();
    }

}
