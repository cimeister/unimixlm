using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PlayerController : MonoBehaviour
{
    public float verticalMovement;
    public float horizontalMovement;
    float minimumDistanceBetweenPlatforms = 0.5f;

    public GameObject cameraObj;
    public GameObject gameOver;
    public GameObject platform;
    public GameObject platformPlacer;
    public Text scoreText;
    int score = 0;

    GameObject lastPlatform;

    const float MIN_X = -1f, MAX_X = 1f;
    //public GameObject gameOver;

    Rigidbody2D rb;
    bool IsFalling { get => rb.velocity.y < 0; }
    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody2D>();

        for (float y = -0.3f; y < 1f; y += 0.04f)
            PlaceRandomPlatform(y);
    }

    // Update is called once per frame
    void Update()
    {
        if (rb.position.x <= MIN_X)
            rb.transform.position = new Vector3(MAX_X - 0.01f, rb.transform.position.y);
        if (rb.position.x >= MAX_X)
            rb.transform.position = new Vector3(MIN_X + 0.01f, rb.transform.position.y);

        if (Input.GetKey(KeyCode.D))
        {
            Debug.Log("move right");
            rb.transform.position += new Vector3(horizontalMovement * Time.deltaTime, 0);
        }
        if (Input.GetKey(KeyCode.A))
        {
            Debug.Log("move left");
            rb.transform.position += new Vector3(-horizontalMovement * Time.deltaTime, 0);
        }
        if (PositionCamera())
            PlaceRandomPlatform();
        UpdateScore();
    }

    private void UpdateScore()
    {
        int newScore = (int)(rb.transform.position.y * 100f);
        if(newScore > score)
        {
            score = newScore;
            scoreText.text = score.ToString();
        }
    }

    bool PositionCamera()
    {
        if (cameraObj.transform.position.y < rb.transform.position.y)
        {
            cameraObj.transform.Translate(new Vector3(0, rb.transform.position.y - cameraObj.transform.position.y, 0));
            return true;
        }
        return false;
    }

    private void OnTriggerEnter2D(Collider2D trigger)
    {
        if (IsFalling && trigger.gameObject.CompareTag("Platform"))
        {
            Debug.LogWarning("platform hit falling");
            rb.AddForce(transform.up * verticalMovement, ForceMode2D.Impulse); // delta time?
        }
        if (trigger.gameObject.CompareTag("DeathZone"))
        {
            Debug.LogWarning("game over");
            Time.timeScale = 0;
            gameOver.SetActive(true);
        }
    }

    void PlaceRandomPlatform(float y = -1000f)
    {
        if (y < -800f) y = platformPlacer.transform.position.y;

        var randX = UnityEngine.Random.Range(MIN_X, MAX_X);

        bool closeToPrev = false;
        var newPos = new Vector3(randX, y);
        if (lastPlatform != null)
            closeToPrev = newPos.y-lastPlatform.transform.position.y<0.01f || ((newPos - lastPlatform.transform.position).magnitude < minimumDistanceBetweenPlatforms);
        

        if (!closeToPrev)
            lastPlatform = Instantiate(platform, newPos, Quaternion.identity);
    }

    public void Restart()
    {
        Debug.Log("retry");
        Time.timeScale = 1;
        UnityEngine.SceneManagement.SceneManager.LoadScene("SampleScene");
    }
}