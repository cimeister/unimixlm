using System;
using Extensions;
using UnityEngine;
using UnityEngine.EventSystems;

namespace Cards.DragDrop
{
    public class DragDrop : MonoBehaviour, IPointerDownHandler, IBeginDragHandler, IEndDragHandler, IDragHandler
    {
        [SerializeField] protected float dragAlpha = 0.7f;
        [ReadOnly] [SerializeField] protected bool dragging;
        [ReadOnly] [SerializeField] protected bool hadValidDrop;
        [ReadOnly] [SerializeField] protected Vector3 dragStartPosition;

        protected CanvasGroup canvasGroup;

        protected virtual void Awake()
        {
            canvasGroup = GetComponent<CanvasGroup>();
        }

        public virtual void OnPointerDown(PointerEventData eventData)
        {
        }

        public virtual void OnBeginDrag(PointerEventData eventData)
        {
            dragStartPosition = transform.position;

            hadValidDrop = false;
            dragging = true;
            canvasGroup.blocksRaycasts = false;
            canvasGroup.alpha = dragAlpha;
        }

        public virtual void OnEndDrag(PointerEventData eventData)
        {
            dragging = false;
            canvasGroup.blocksRaycasts = true;
            canvasGroup.alpha = 1;

            if (!hadValidDrop)
            {
                transform.position = dragStartPosition;
            }

            hadValidDrop = false;
        }

        public virtual void OnDrag(PointerEventData eventData)
        {
            dragging = true;

            Vector3 screenCenter =
                new Vector3(Screen.width * 0.5f, Screen.height * 0.5f, -Camera.main.transform.position.z);
            Vector3 screenTouch = screenCenter + new Vector3(eventData.delta.x, eventData.delta.y, 0);

            Vector3 worldCenterPosition = Camera.main.ScreenToWorldPoint(screenCenter);
            Vector3 worldTouchPosition = Camera.main.ScreenToWorldPoint(screenTouch);

            Vector3 delta = worldTouchPosition - worldCenterPosition;
            transform.position += delta;
        }

        public virtual void SetValidDrop(int x, int y, Vector3? pos = null)
        {
            hadValidDrop = true;
            if (pos != null)
            {
                transform.position = pos.Value;
            }
        }
    }
}