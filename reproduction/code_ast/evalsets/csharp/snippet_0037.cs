using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Exrin.Abstraction
{
    /// <summary>
    /// An execution entity to perform when an action is taken in the ViewModel
    /// </summary>
    public interface IViewModelExecute
    {
        /// <summary>
        /// A list of operations to perform in sequence and their associated rollbacks
        /// </summary>
        List<IBaseOperation> Operations { get; }
       
        /// <summary>
        /// The total length allowed for the operations to complete, before a timeout is triggered. Setting to 0 or below will result in no timeout constraint.
        /// </summary>
        int TimeoutMilliseconds { get; }

    }
}