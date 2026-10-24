using System;
using System.Collections.Generic;
using System.Text;

namespace PipServices3.Components.Trace
{
    /// <summary>
    /// Dummy implementation of tracer that doesn't do anything.
    /// It can be used in testing or in situations when tracing is required
    /// but shall be disabled.
    /// 
    /// See <see cref="ITracer"/>
    /// </summary>
    public class NullTracer : ITracer
    {
        /// <summary>
        /// Creates a new instance of the tracer.
        /// </summary>
        public NullTracer() { }

        /// <summary>
        /// Begings recording an operation trace
        /// </summary>
        /// <param name="correlationId">(optional) transaction id to trace execution through call chain.</param>
        /// <param name="component">a name of called component</param>
        /// <param name="operation">a name of the executed operation. </param>
        /// <returns>a trace timing object.</returns>
        public TraceTiming BeginTrace(string correlationId, string component, string operation)
        {
            return new TraceTiming(correlationId, component, operation, this);
        }

        /// <summary>
        /// Records an operation failure with its name, duration and error
        /// </summary>
        /// <param name="correlationId"> (optional) transaction id to trace execution through call chain.</param>
        /// <param name="component">a name of called component</param>
        /// <param name="operation">a name of the executed operation. </param>
        /// <param name="error">an error object associated with this trace.</param>
        /// <param name="duration">execution duration in milliseconds. </param>
        public void Failure(string correlationId, string component, string operation, Exception error, long duration)
        {
            // Do nothing...
        }

        /// <summary>
        /// Records an operation trace with its name and duration
        /// </summary>
        /// <param name="correlationId">(optional) transaction id to trace execution through call chain.</param>
        /// <param name="component">a name of called component</param>
        /// <param name="operation">a name of the executed operation. </param>
        /// <param name="duration">execution duration in milliseconds. </param>
        public void Trace(string correlationId, string component, string operation, long duration)
        {
            // Do nothing...
        }
    }
}