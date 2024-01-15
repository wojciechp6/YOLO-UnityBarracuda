using NN;
using NUnit.Framework;
using NUnit.Framework.Internal;
using System.Collections.Generic;
using UnityEngine;

namespace Tests
{
    public class TestDuplicatesSupressor
    {
        [Test]
        public void ShouldZeroSecondBoxWhenInOrder()
        {
            List<ResultBox> boxes = new();

            int bestClass = 1;
            float box1ClassScore = 0.9f;
            float box2ClassScore = 0.7f;
            ResultBox box1 = CreateTestResultBox(bestClass, box1ClassScore);
            ResultBox box2 = CreateTestResultBox(bestClass, box2ClassScore);
            boxes.Add(box1);
            boxes.Add(box2);

            DuplicatesSupressor.RemoveDuplicats(boxes);

            Assert.AreEqual(box1ClassScore, box1.classes[bestClass]);
            Assert.AreEqual(0, box2.classes[bestClass]);
        }

        [Test]
        public void ShouldZeroFirstBoxWhenOutOfOrder()
        {
            List<ResultBox> boxes = new();

            int bestClass = 1;
            float box1ClassScore = 0.7f;
            float box2ClassScore = 0.9f;
            ResultBox box1 = CreateTestResultBox(bestClass, box1ClassScore);
            ResultBox box2 = CreateTestResultBox(bestClass, box2ClassScore);
            boxes.Add(box1);
            boxes.Add(box2);

            DuplicatesSupressor.RemoveDuplicats(boxes);

            Assert.AreEqual(0, box1.classes[bestClass]);
            Assert.AreEqual(box2ClassScore, box2.classes[bestClass]);
        }

        [Test]
        public void ShouldNotZeroWhenDifferentClasses()
        {
            List<ResultBox> boxes = new();

            int box1BestClass = 1;
            int box2BestClass = 5;
            float box1ClassScore = 0.7f;
            float box2ClassScore = 0.9f;
            ResultBox box1 = CreateTestResultBox(box1BestClass, box1ClassScore);
            ResultBox box2 = CreateTestResultBox(box2BestClass, box2ClassScore);
            boxes.Add(box1);
            boxes.Add(box2);

            DuplicatesSupressor.RemoveDuplicats(boxes);

            Assert.AreEqual(box1ClassScore, box1.classes[box1BestClass]);
            Assert.AreEqual(box2ClassScore, box2.classes[box2BestClass]);
        }

        [Test]
        public void ShouldZeroWhenCloseRects()
        {
            List<ResultBox> boxes = new();

            int bestClass = 1;
            float box1ClassScore = 0.7f;
            float box2ClassScore = 0.9f;
            Rect box1Rect = new(0.1f, 0.1f, 0.9f, 0.9f);
            Rect box2Rect = new(0.11f, 0.12f, 0.88f, 0.87f);
            ResultBox box1 = CreateTestResultBox(bestClass, box1ClassScore, box1Rect);
            ResultBox box2 = CreateTestResultBox(bestClass, box2ClassScore, box2Rect);
            boxes.Add(box1);
            boxes.Add(box2);

            DuplicatesSupressor.RemoveDuplicats(boxes);

            Assert.AreEqual(0, box1.classes[bestClass]);
            Assert.AreEqual(box2ClassScore, box2.classes[bestClass]);
        }

        ResultBox CreateTestResultBox(int bestClass, float classScore, Rect rect)
        {
            const int classesNum = 20;
            float[] classes = new float[classesNum];
            classes[bestClass] = classScore;

            ResultBox box = new ResultBox
            {
                bestClassIndex = bestClass,
                classes = classes,
                rect = rect
            };
            return box;
        }

        ResultBox CreateTestResultBox(int bestClass, float classScore)
        {
            Rect rect = new(0, 0, 1, 1);
            return CreateTestResultBox(bestClass, classScore, rect);
        }
    }
}
