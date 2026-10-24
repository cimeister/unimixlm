using UnityEngine;

namespace Metro.Utilities
{
    public class AutoRotation : MonoBehaviour
    {
        public enum AxisType
        {
            X = 0,
            Y = 1,
            Z = 2,
        }

        [SerializeField]
        private AxisType axisType;
        
        [SerializeField]
        private float speed;

        private Vector3 cachedDirection;

        private Transform cachedTransform;

        void Awake()
        {
            switch (this.axisType)
            {
                case AxisType.X:
                    this.cachedDirection = Vector3.right;
                    break;
                case AxisType.Y:
                    this.cachedDirection = Vector3.up;
                    break;
                case AxisType.Z:
                    this.cachedDirection = Vector3.forward;
                    break;
            }

            this.cachedTransform = this.transform;
        }

        void Update()
        {
            this.cachedTransform.localRotation *= Quaternion.AngleAxis(this.speed * Time.deltaTime, this.cachedDirection);
        }
    }
}