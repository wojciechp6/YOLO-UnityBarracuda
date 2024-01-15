using Unity.Barracuda;
using UnityEngine;

namespace NN
{
    public class ResultBox
    {
        public Rect rect;
        public float score;
        public float[] classes;
        public int bestClassIndex;
    }
}