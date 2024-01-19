using Unity.Barracuda;
using UnityEngine;

namespace NN
{
    public class ResultBox
    {
        public Rect rect;
        public float score;
        public int bestClassIndex;
    }
}