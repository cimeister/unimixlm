package com.code.street;

import java.util.*;
import java.io.*;
import java.math.*;

class PowerOfThorEpisode1 {

    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        int lightX = in.nextInt(); // the X position of the light of power
        int lightY = in.nextInt(); // the Y position of the light of power
        int initialTX = in.nextInt(); // Thor's starting X position
        int initialTY = in.nextInt(); // Thor's starting Y position

        // game loop
        while (true) {
            int remainingTurns = in.nextInt(); // The remaining amount of turns Thor can move. Do not remove this line.
            String directionX = "";
            String directionY = "";
            if (initialTX > lightX) {
                directionX = "W";
                initialTX--;
            }
            if (initialTX < lightX) {
                directionX = "E";
                initialTX++;
            }

            if (initialTY > lightY) {
                directionY = "N";
                initialTY--;
            }
            if (initialTY < lightY) {
                directionY = "S";
                initialTY++;
            }

            // To debug: System.err.println("Debug messages...");

            System.out.println(directionY + directionX); // A single line providing the move to be made: N NE E SE S SW W or NW           // Write an action using System.out.println()
            // To debug: System.err.println("Debug messages...");
        }
    }
}