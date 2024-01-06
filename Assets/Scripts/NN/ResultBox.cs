using UnityEngine;

namespace NN
{
    public struct ResultBox
    {
        public Rect rect;
        public float confidence;
        public float[] classes;
        public int bestClassIdx;
    }
}