using System.Collections.Generic;
using UnityEngine;

public class OnGUICanvasRelativeDrawer : MonoBehaviour
{
    struct Label
    {
        public string text;
        public Rect rect;
    }

    [HideInInspector]
    public RectTransform relativeObject { set { relative = value; rootCanvas = GetRootCanvas(value); } }
    RectTransform relative;
    RectTransform rootCanvas;

    List<Label> labels = new List<Label>();
    GUIStyle style;

    /// <summary>
    /// Draws label on screen above relative object
    /// </summary>
    /// <param name="text">Text to draw</param>
    /// <param name="position">Normalized position of label</param>
    public void DrawLabel(string text, Vector2 position)
    {
        Vector2 anchor = GetAnchorPosition();
        Rect rect = new Rect(anchor + position * relative.rect.size, new Vector2(150, 100));
        labels.Add(new Label { text = text, rect = rect });
    }

    /// <summary>
    /// Remove all previous draws
    /// </summary>
    public void Clear()
    {
        labels.Clear();
    }

    private GUIStyle GetStyle()
    {
        return new GUIStyle { fontSize = 20, normal = new GUIStyleState { textColor = Color.white } };
    }

    private void OnGUI()
    {
        style = style != null ? style : GetStyle();
        foreach (var label in labels)
        {
            GUI.Label(label.rect, label.text, style);
        }
    }

    private Vector2 GetAnchorPosition()
    {
        return new Vector2(relative.localPosition.x, -relative.localPosition.y) + rootCanvas.rect.size / 2 - relative.rect.size / 2;
    }

    private RectTransform GetRootCanvas(RectTransform rectTransform)
    {
        Transform parent = rectTransform.transform;
        while (true)
        {
            if (parent.parent != null && parent.parent.GetComponent<RectTransform>() != null)
                parent = parent.parent;
            else
                return parent.GetComponent<RectTransform>();
        }
    }
}
