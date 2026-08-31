using System;

namespace Octgn.Communication
{
    public sealed class ErrorResponseException : Exception
    {
        public string Code { get; set; }
        public bool IsCritical { get; set; }

        public ErrorResponseException(string code, string message, bool isCritical) : this(message) {
            IsCritical = isCritical;
            Code = code;
        }

        public ErrorResponseException(ErrorResponseData data)
            : this(data.Code, data.Message, data.IsCritical) {
        }

        public ErrorResponseException() : base() {
        }

        public ErrorResponseException(string message) : base(message) {
        }

        public ErrorResponseException(string message, Exception innerException) : base(message, innerException) {
        }
    }
}