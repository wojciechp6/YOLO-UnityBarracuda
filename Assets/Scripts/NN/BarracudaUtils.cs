using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class BarracudaUtils  
{
    public static IOps CreateOps(WorkerFactory.Type type, bool verbose = false)
    {
        WorkerFactory.ValidateType(type);
        switch (type)
        {
            case WorkerFactory.Type.ComputePrecompiled:
                return new PrecompiledComputeOps(ComputeShaderSingleton.Instance.kernels,
                                                ComputeShaderSingleton.Instance.referenceKernels, verbose: verbose);

            case WorkerFactory.Type.Compute:
                return new ComputeOps(ComputeShaderSingleton.Instance.kernels,
                                     ComputeShaderSingleton.Instance.referenceKernels, verbose: verbose);

            case WorkerFactory.Type.ComputeRef:
                return new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels);

            case WorkerFactory.Type.CSharp:
                return new UnsafeArrayCPUOps();

            default:
                return new ReferenceCPUOps();
        }
    }
}
