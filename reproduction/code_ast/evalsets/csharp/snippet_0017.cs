using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class TurretController : NetworkBehaviour
{
    public PivotGun pivotGun;
    public PivotTurret pivotTurret;

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (!isLocalPlayer)
        {
            return;
        }
		if (!Input.GetKey(KeyCode.Space))
        {
            pivotGun.Update();
            pivotTurret.Update();
        }
    }
}