using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class PerformanceCounter : MonoBehaviour
{
    public class Counter
    {
        protected string name;
        protected float value;

        public float Value { get => value; set => this.value = value; }
        public string Name { get => name; set => name = value; }

        public Counter(string name, float value)
        {
            this.Name = name;
            this.Value = value;
        }
    }

    public class StopwatchCounter : Counter
    {
        public new float Value { get => value; }

        Stopwatch stopwatch;

        public StopwatchCounter(string name) : base(name, 0)
        {
            this.name = name;
            stopwatch = new Stopwatch();
        }

        public void Start()
        { stopwatch.Restart(); }

        public void Stop()
        {
            stopwatch.Stop();
            value = stopwatch.ElapsedMilliseconds;
        }
    }

    static PerformanceCounter firstInstance;

    public Text textField;
    public bool showFPS = true;
    [Range(1, 250)]
    public float FPSSmothness = 33;

    List<Counter> counters = new List<Counter>();
    float previousFPS;

    StringBuilder builder = new StringBuilder();

    void Start()
    {
        if (firstInstance == null)
            firstInstance = this;
    }

    void Update()
    {
        if (showFPS)
        {
            builder.Append(GetSmoothFPS());
            builder.Append(" FPS");
            builder.AppendLine();
        }

        foreach (var counter in counters)
        {
            builder.Append(counter.Name);
            builder.Append(": ");
            builder.Append(counter.Value);
            builder.AppendLine();
        }

        textField.text = builder.ToString();
        builder.Clear();
    }

    /// <summary>
    /// Returns first instance of PerformanceCounter
    /// </summary>
    public static PerformanceCounter GetInstance()
    {
        return firstInstance;
    }

    public Counter AddCounter(string name)
    {
        var c = new Counter(name, 0);
        counters.Add(c);
        return c;
    }

    public void AddCounter(Counter counter)
    {
        counters.Add(counter);
    }

    private int GetSmoothFPS()
    {
        float fps = 1f / Time.unscaledDeltaTime;
        previousFPS = (previousFPS + fps * (1 / FPSSmothness)) / (1 + (1 / FPSSmothness));
        return (int)previousFPS;
    }


}
