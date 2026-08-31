using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SnakeMovement : MonoBehaviour
{
    [SerializeField]Vector3 moveDirection;
    public Vector3 currentDirection;
    [SerializeField]float timePassed;
    float maxTime = 0.2f;
    public GameObject bodyPrefab;
    [SerializeField]List<GameObject> bodyTiles = new List<GameObject>();
    public bool stop;

    void Start()
    {
        for (int i = 0; i < 9; i++)
        {
            GameObject tile = Instantiate(bodyPrefab, transform.position - new Vector3((i + 1) * transform.localScale.x, 0), Quaternion.identity);
            tile.transform.localScale = transform.localScale;
            bodyTiles.Add(tile);
        }
    }

    void Update()
    {
        //if(transform.position.x < 95.2 || transform.position.x > 104.8 ||
        //   transform.position.y < -3.3 || transform.position.y > 3.3)
        //{
        //    stop = true;
        //    GameOver();
        //}
            
        if(!stop)
        {
            if (Input.GetKeyDown(KeyCode.UpArrow) && currentDirection.y == 0)
            {
                moveDirection = new Vector3(0, 1);
            }
            else if (Input.GetKeyDown(KeyCode.DownArrow) && currentDirection.y == 0)
            {
                moveDirection = new Vector3(0, -1);
            }
            else if (Input.GetKeyDown(KeyCode.RightArrow) && currentDirection.x == 0)
            {
                moveDirection = new Vector3(1, 0);
            }
            else if (Input.GetKeyDown(KeyCode.LeftArrow) && currentDirection.x == 0)
            {
                moveDirection = new Vector3(-1, 0);
            }

            timePassed += Time.deltaTime;

            if (timePassed >= maxTime && moveDirection != Vector3.zero)
            {
                currentDirection = moveDirection;
                for (int i = 0; i < bodyTiles.Count; i++)
                {
                    if (bodyTiles[i].transform.position == transform.position + currentDirection * transform.localScale.x)
                    {
                        GameOver();
                        break;
                    }
                }
                if(transform.position.x + currentDirection.x * transform.localScale.x < -5.1 || transform.position.x + currentDirection.x * transform.localScale.x > 5.1 ||
                   transform.position.y + currentDirection.y * transform.localScale.x < -3.6 || transform.position.y + currentDirection.y * transform.localScale.x > 3.6)
                {
                    GameOver();
                }
                if (!stop)
                {
                    
                    timePassed = 0;
                    GameObject tile = Instantiate(bodyPrefab, transform.position, Quaternion.identity);
                    tile.transform.localScale = transform.localScale;
                    bodyTiles.Insert(0, tile);
                    Destroy(bodyTiles[bodyTiles.Count - 1]);
                    bodyTiles.RemoveAt(bodyTiles.Count - 1);    
                    transform.position += currentDirection * transform.localScale.x;
                    if(currentDirection.y == 1)
                    {
                        transform.rotation = Quaternion.Euler(0, 0, 90);
                    }
                    else if (currentDirection.y == -1)
                    {
                        transform.rotation = Quaternion.Euler(0, 0, -90);
                    }
                    else if (currentDirection.x == 1)
                    {
                        transform.rotation = Quaternion.Euler(0, 0, 0);
                    }
                    else if (currentDirection.x == -1)
                    {
                        transform.rotation = Quaternion.Euler(0, 0, -180);
                    }

                }

            }
        }

        void GameOver()
        {
            Debug.Log("Game Over :(");
            stop = true;
        }
    }
}