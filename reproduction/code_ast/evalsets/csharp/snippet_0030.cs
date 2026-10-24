using System.Collections;
using System.Collections.Generic;
using UnityEngine.Tilemaps;
using UnityEngine;

public class TowerSelector : MonoBehaviour
{
    public Tilemap tilemap;
    public SpriteRenderer arrows;
    public SpriteRenderer towerPreview;
    public List<Transform> towers;
    

    Vector3 selectedTilePos;
    int selectedTowerIndex;
    int showingTowerIndex;
    bool showingPreview;
    Vector3 maxVector;
    TileHighlight tileHighlight;

    void Start()
    {
        selectedTowerIndex = 0;
        showingTowerIndex = 0;
        showingPreview = false;
        towerPreview.sprite = towers[selectedTowerIndex].GetComponent<SpriteRenderer>().sprite;
        towerPreview.color = new Color(1f,1f,1f,0.7f);
        maxVector = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        selectedTilePos = maxVector;
        tileHighlight = GetComponent<TileHighlight>();
    }

    // Update is called once per frame
    void Update()
    {
        /*
         * Input listeners
         */

        // Register selected tile position on primary click
        // Add tower if same cell is clicked twice
        if (Input.GetMouseButtonDown(0)) 
        {
            tileHighlight.CallUpdate();

            //Get tile postition
            Vector3 cellPos = GetCellPosAtMouse();

            // Add tower if same cell last clicked
            if(selectedTilePos == cellPos) 
            {
                AddTower(selectedTilePos, selectedTowerIndex);
                selectedTilePos = maxVector;
            }
            else 
            {
                // Reset tower index and preview
                selectedTowerIndex = 0;
                showingPreview = false;

                // Register tile postition
                if (tilemap.HasTile(tilemap.WorldToCell(cellPos)))
                {
                    selectedTilePos = cellPos;
                }
            }
            
        }
        // Unregister selected tile position on secondary click
        else if (Input.GetMouseButtonDown(1))
        {
            selectedTilePos = maxVector;
        }
        // Increment selected tower index on scroll up
        else if (Input.GetAxis("Mouse ScrollWheel") > 0f)
        {
            if(selectedTowerIndex < towers.Count - 1)
            {
                selectedTowerIndex++;
            }
            else 
            {
                selectedTowerIndex = 0;    
            }
        }
        // Decrement selected tower index on scroll down
        else if (Input.GetAxis("Mouse ScrollWheel") < 0f)
        {
            if(selectedTowerIndex > 0)
            {
                selectedTowerIndex--;
            }
            else 
            {
                selectedTowerIndex = towers.Count - 1;    
            }
        }
    
        /*
         * Selection rendering
        */
        // Show preview on selected tile
        if(selectedTilePos != maxVector && !showingPreview)
        {
            showingPreview = true;
            arrows.transform.position = selectedTilePos;
            towerPreview.transform.position = selectedTilePos;
            arrows.enabled = true;
            towerPreview.enabled = true;
        }
        // Remove preview when deselected
        else if(selectedTilePos == maxVector && showingPreview)
        {
            arrows.enabled = false;
            towerPreview.enabled = false;
            showingPreview = false;
        }
        // Change preview when selected tower changes
        if(selectedTilePos != maxVector && showingPreview)
        {
            if(showingTowerIndex != selectedTowerIndex) 
            {
                towerPreview.sprite = towers[selectedTowerIndex].GetComponent<SpriteRenderer>().sprite;
                showingTowerIndex = selectedTowerIndex;
            }
        }
        // Highlight on hover
        if(selectedTilePos == maxVector)
        {
            tileHighlight.CallUpdate();
        }

    }

    Vector3 GetCellPosAtMouse() 
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        Vector3 worldPos = ray.GetPoint(-ray.origin.z / ray.direction.z);
        return tilemap.GetCellCenterWorld(tilemap.WorldToCell(worldPos));
    }

    void AddTower(Vector3 pos, int towerIndex)
    {
        Instantiate(towers[towerIndex], pos, Quaternion.identity);
    }
}