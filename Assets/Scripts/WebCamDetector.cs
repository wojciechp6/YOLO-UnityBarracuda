using Assets.Scripts;
using NN;
using System;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.UI;

[RequireComponent(typeof(OnGUICanvasRelativeDrawer))]
public class WebCamDetector : MonoBehaviour
{
    [Tooltip("File of YOLO model. If you want to use another than YOLOv2 tiny, it may be necessary to change some const values in YOLOHandler.cs")]
    public NNModel ModelFile;
    [Tooltip("Text file with classes names separated by coma ','")]
    public TextAsset ClassesTextFile;

    [Tooltip("RawImage component which will be used to draw resuls.")]
    public RawImage ImageUI;

    [Range(0.0f, 1f)]
    [Tooltip("The minimum value of box confidence below which boxes won't be drawn.")]
    public float MinBoxConfidence = 0.3f;

    NNHandler nn;
    YOLOHandler yolo;

    WebCamTextureProvider CamTextureProvider;

    string[] classesNames;
    OnGUICanvasRelativeDrawer relativeDrawer;

    Color[] colorArray = new Color[] { Color.red, Color.green, Color.blue, Color.cyan, Color.magenta, Color.yellow };

    void OnEnable()
    {
        nn = new NNHandler(ModelFile);
        yolo = new YOLOHandler(nn);

        var firstInput = nn.model.inputs[0];
        int height = firstInput.shape[5];
        int width = firstInput.shape[6];

        CamTextureProvider = new WebCamTextureProvider(width, height);
        CamTextureProvider.Start();

        relativeDrawer = GetComponent<OnGUICanvasRelativeDrawer>();
        relativeDrawer.relativeObject = ImageUI.GetComponent<RectTransform>();

        classesNames = ClassesTextFile.text.Split(',');
        YOLOv2Postprocessor.DiscardThreshold = MinBoxConfidence;
    }

    void Update()
    {
        Texture2D texture = GetNextTexture();

        var boxes = yolo.Run(texture);
        DrawResults(boxes, texture);
        ImageUI.texture = texture;
    }

    Texture2D GetNextTexture()
    {
        return CamTextureProvider.GetTexture();
    }

    private void OnDisable()
    {
        nn.Dispose();
        yolo.Dispose();
        CamTextureProvider.Stop();
    }

    private void DrawResults(IEnumerable<ResultBox> results, Texture2D texture)
    {
        relativeDrawer.Clear();
        foreach(ResultBox box in results)
            DrawBox(box, texture);
    }

    private void DrawBox(ResultBox box, Texture2D img)
    {
        if (box.score < MinBoxConfidence)
            return;

        Color boxColor = colorArray[box.bestClassIndex % colorArray.Length];
        int boxWidth = (int)(box.score / MinBoxConfidence);
        TextureTools.DrawRectOutline(img, box.rect, boxColor, boxWidth, rectIsNormalized: false, revertY: true);

        Vector2 textureSize = new(img.width, img.height);
        relativeDrawer.DrawLabel(classesNames[box.bestClassIndex], box.rect.position / textureSize);
    }
}
