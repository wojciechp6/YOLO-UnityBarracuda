using UnityEngine;
using System.Collections;
using Unity.Barracuda;

public class NNHandler 
{
    public Model model;
    public IWorker worker;

    public NNHandler(NNModel nnmodel)
    {
        model = ModelLoader.Load(nnmodel);
        worker = WorkerFactory.CreateWorker(model);
    }
}
