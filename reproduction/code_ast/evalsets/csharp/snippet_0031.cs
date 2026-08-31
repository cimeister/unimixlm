using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpriteReTexture : MonoBehaviour
{
    [Range(0.4f, 0.6f)]
    [SerializeField] private float kValue;

    float kValuePrior;
    
    void Start()
    {
        kValuePrior = kValue;

        CreateTexture();
    }

    private void Update()
    {
        if(kValue != kValuePrior)
        {
            kValuePrior = kValue;
            CreateTexture();
        }
    }

    private void CreateTexture()
    {
        Texture2D texture = new Texture2D(400, 400);
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, 400, 400), Vector2.zero);
        GetComponent<SpriteRenderer>().sprite = sprite;

        float xValue1 = 0.00001f;
        float xValue2;
        float xValue3;

        for (int y = 0; y < texture.height; y++)
        {
            xValue2 = kValue * xValue1 * (1 - xValue1);
            xValue3 = kValue * xValue2 * (1 - xValue2);
            xValue1 = kValue * xValue3 * (1 - xValue3);

            xValue2 = xValue1 * xValue1 - xValue1 - kValue;
            xValue3 = xValue2 * xValue2 - xValue2 - kValue;
            xValue1 = xValue3 * xValue3 - xValue3 - kValue;

            for (int x = 0; x < texture.width; x++) //Goes through each pixel
            {
                Color pixelColour = new Color(Mathf.Abs(xValue1), Mathf.Abs(xValue2), Mathf.Abs(xValue3), 1);

                texture.SetPixel(x, y, pixelColour);
            }
        }
        texture.Apply();
    }
}