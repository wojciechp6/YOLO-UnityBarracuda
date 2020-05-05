using System;
using UnityEngine;


public class TextureTransformTools
{
    // Based on https://gist.github.com/natsupy/e129936543f9b4663a37ea0762172b3b

    public static Texture2D WebCamTextureToTexture2D(WebCamTexture webCamTexture)
    {
        var tex = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.ARGB32, false);
        tex.SetPixels(webCamTexture.GetPixels());
        tex.Apply();
        return tex;
    }

    public static Texture2D CropToSquare(Texture2D tex)
    {
        var smaller = tex.width < tex.height ? tex.width : tex.height;
        var xOffset = (tex.width - smaller) / 2;
        var yOffset = (tex.height - smaller) / 2;
        return CropWithRect(tex, new Rect(xOffset, yOffset, smaller, smaller));
    }

    public static Texture2D CropToSquare(WebCamTexture tex)
    {
        var smaller = tex.width < tex.height ? tex.width : tex.height;
        return CropWithRect(tex, new Rect(0, 0, smaller, smaller));
    }

    public static void CropToSquare(WebCamTexture tex, ref Texture2D outputTexture)
    {
        var smaller = tex.width < tex.height ? tex.width : tex.height;
        CropWithRect(tex, ref outputTexture, new Rect(0, 0, smaller, smaller));
    }

    public static Texture2D CropWithRect(Texture2D texture, Rect rect)
    {
        if (rect.height < 0 || rect.width < 0)
        {
            throw new System.ArgumentException("Invalid texture size");
        }

        Texture2D result = new Texture2D((int)rect.width, (int)rect.height);

        if (rect.width != 0 && rect.height != 0)
        {
            float xRect = rect.x;
            float yRect = rect.y;
            float widthRect = rect.width;
            float heightRect = rect.height;

            xRect = (texture.width - rect.width) / 2;
            yRect = (texture.height - rect.height) / 2;

            if (texture.width < rect.x + rect.width || texture.height < rect.y + rect.height ||
                xRect > rect.x + texture.width || yRect > rect.y + texture.height ||
                xRect < 0 || yRect < 0 || rect.width < 0 || rect.height < 0)
            {
                throw new System.ArgumentException("Set value crop less than origin texture size");
            }

            result.SetPixels(texture.GetPixels(Mathf.FloorToInt(xRect), Mathf.FloorToInt(yRect),
                                            Mathf.FloorToInt(widthRect), Mathf.FloorToInt(heightRect)));
            result.Apply();
        }

        return result;
    }

    public static Texture2D CropWithRect(WebCamTexture texture, Rect rect)
    {
        if (rect.height < 0 || rect.width < 0)
        {
            throw new System.ArgumentException("Invalid texture size");
        }

        Texture2D result = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);

        if (rect.width != 0 && rect.height != 0)
        {
            float xRect = rect.x;
            float yRect = rect.y;
            float widthRect = rect.width;
            float heightRect = rect.height;

            xRect = (texture.width - rect.width) / 2;
            yRect = (texture.height - rect.height) / 2;

            if (texture.width < rect.x + rect.width || texture.height < rect.y + rect.height ||
                xRect > rect.x + texture.width || yRect > rect.y + texture.height ||
                xRect < 0 || yRect < 0 || rect.width < 0 || rect.height < 0)
            {
                throw new System.ArgumentException("Set value crop less than origin texture size");
            }
            result.SetPixels(texture.GetPixels(Mathf.FloorToInt(xRect), Mathf.FloorToInt(yRect),
                                            Mathf.FloorToInt(widthRect), Mathf.FloorToInt(heightRect)));
            result.Apply();
        }

        return result;
    }

    public static void CropWithRect(WebCamTexture texture, ref Texture2D outputTexture, Rect rect)
    {
        if (rect.height < 0 || rect.width < 0)
        {
            throw new System.ArgumentException("Invalid texture size");
        }

        if (outputTexture == null)
            outputTexture = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);

        if (outputTexture.width != rect.width || outputTexture.height != rect.height)
            outputTexture.Resize((int)rect.width, (int)rect.height);

        Texture2D result = outputTexture;

        if (rect.width != 0 && rect.height != 0)
        {
            float xRect = rect.x;
            float yRect = rect.y;
            float widthRect = rect.width;
            float heightRect = rect.height;

            xRect = (texture.width - rect.width) / 2;
            yRect = (texture.height - rect.height) / 2;

            if (texture.width < rect.x + rect.width || texture.height < rect.y + rect.height ||
                xRect > rect.x + texture.width || yRect > rect.y + texture.height ||
                xRect < 0 || yRect < 0 || rect.width < 0 || rect.height < 0)
            {
                throw new System.ArgumentException("Set value crop less than origin texture size");
            }
            result.SetPixels(texture.GetPixels(Mathf.FloorToInt(xRect), Mathf.FloorToInt(yRect),
                                            Mathf.FloorToInt(widthRect), Mathf.FloorToInt(heightRect)));
            result.Apply();
        }
    }

    /// <summary>
    ///     Returns a scaled copy of given texture.
    /// </summary>
    /// <param name="tex">Source texure to scale</param>
    /// <param name="width">Destination texture width</param>
    /// <param name="height">Destination texture height</param>
    /// <param name="mode">Filtering mode</param>
    public static Texture2D scaled(Texture2D src, int width, int height, FilterMode mode = FilterMode.Trilinear)
    {
        Rect texR = new Rect(0, 0, width, height);
        _gpu_scale(src, width, height, mode);

        //Get rendered data back to a new texture
        Texture2D result = new Texture2D(width, height, TextureFormat.ARGB32, true);
        result.Resize(width, height);
        result.ReadPixels(texR, 0, 0, true);
        return result;
    }

    /// <summary>
    /// Scales the texture data of the given texture.
    /// </summary>
    /// <param name="tex">Texure to scale</param>
    /// <param name="width">New width</param>
    /// <param name="height">New height</param>
    /// <param name="mode">Filtering mode</param>
    public static void scale(Texture2D tex, int width, int height, FilterMode mode = FilterMode.Trilinear)
    {
        Rect texR = new Rect(0, 0, width, height);
        _gpu_scale(tex, width, height, mode);

        // Update new texture
        tex.Resize(width, height);
        tex.ReadPixels(texR, 0, 0, true);
        tex.Apply(true);        //Remove this if you hate us applying textures for you :)
    }

    // Internal unility that renders the source texture into the RTT - the scaling method itself.
    static void _gpu_scale(Texture2D src, int width, int height, FilterMode fmode)
    {
        //We need the source texture in VRAM because we render with it
        src.filterMode = fmode;
        src.Apply(true);

        //Using RTT for best quality and performance. Thanks, Unity 5
        RenderTexture rtt = new RenderTexture(width, height, 32);

        //Set the RTT in order to render to it
        Graphics.SetRenderTarget(rtt);

        //Setup 2D matrix in range 0..1, so nobody needs to care about sized
        GL.LoadPixelMatrix(0, 1, 1, 0);

        //Then clear & draw the texture to fill the entire RTT.
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, 1, 1), src);
    }


    public static Texture2D RotateTexture(Texture2D originTexture, int angle)
    {
        var result = RotateImageMatrix(
            originTexture.GetPixels32(), originTexture.width, originTexture.height, angle);
        var resultTexture = new Texture2D(originTexture.width, originTexture.height);
        resultTexture.SetPixels32(result);
        resultTexture.Apply();

        return resultTexture;
    }


    public static Color32[] RotateImageMatrix(Color32[] matrix, int width, int height, int angle)
    {
        Color32[] pix1 = new Color32[matrix.Length];

        int x = 0;
        int y = 0;

        Color32[] pix3 = rotateSquare(
            matrix, width, height, (Math.PI / 180 * (double)angle));

        for (int j = 0; j < height; j++)
        {
            for (var i = 0; i < width; i++)
            {
                pix1[x + i + width * (j + y)] = pix3[i + j * width];
            }
        }

        return pix3;
    }


    static Color32[] rotateSquare(Color32[] arr, int width, int height, double phi)
    {
        int x;
        int y;
        int i;
        int j;
        double sn = Math.Sin(phi);
        double cs = Math.Cos(phi);
        Color32[] arr2 = new Color32[arr.Length];

        int xc = width / 2;
        int yc = height / 2;

        for (j = 0; j < height; j++)
        {
            for (i = 0; i < width; i++)
            {
                arr2[j * width + i] = new Color32(0, 0, 0, 0);
                x = (int)(cs * (i - xc) + sn * (j - yc) + xc);
                y = (int)(-sn * (i - xc) + cs * (j - yc) + yc);
                if ((x > -1) && (x < width) && (y > -1) && (y < height))
                {
                    arr2[j * width + i] = arr[y * width + x];
                }
            }
        }
        return arr2;
    }
}
