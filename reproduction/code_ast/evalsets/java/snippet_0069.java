/*
 * Copyright (c) 2014. <NAME>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.gamestudio24.martianrun.actors;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.math.Rectangle;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.gamestudio24.martianrun.enums.GameState;
import com.gamestudio24.martianrun.utils.AssetsManager;
import com.gamestudio24.martianrun.utils.Constants;
import com.gamestudio24.martianrun.utils.GameManager;

public class Background extends Actor {

    private final TextureRegion textureRegion;
    private Rectangle textureRegionBounds1;
    private Rectangle textureRegionBounds2;
    private int speedPercentage = 100;
    private float speed = 10;

    public Background(String assetId, int speedPercentage) {
        this.speedPercentage = speedPercentage;
        textureRegion = AssetsManager.getTextureRegion(assetId);
        textureRegionBounds1 = new Rectangle(0, 0, Constants.APP_WIDTH, Constants.APP_HEIGHT);
        textureRegionBounds2 = new Rectangle(Constants.APP_WIDTH, 0, Constants.APP_WIDTH, Constants.APP_HEIGHT);
    }

    public void setSpeed(float speed) {
        this.speed =  Math.abs(speed);
    }

    @Override
    public void act(float delta) {

        if (GameManager.getInstance().getGameState() != GameState.RUNNING) {
            return;
        }

        if (leftBoundsReached(delta)) {
            resetBounds();
            updateXBounds(-delta);
        } else {
            updateXBounds(-delta);
        }
    }

    @Override
    public void draw(Batch batch, float parentAlpha) {
        super.draw(batch, parentAlpha);
        batch.draw(textureRegion, textureRegionBounds1.x, textureRegionBounds1.y, Constants.APP_WIDTH,
                Constants.APP_HEIGHT);
        batch.draw(textureRegion, textureRegionBounds2.x, textureRegionBounds2.y, Constants.APP_WIDTH,
                Constants.APP_HEIGHT);
    }

    private boolean leftBoundsReached(float delta) {
        return (textureRegionBounds2.x - (delta * transformToScreen(speed / 100 * speedPercentage))) <= 0;
    }

    private void updateXBounds(float delta) {
        textureRegionBounds1.x += delta * transformToScreen(speed / 100 * speedPercentage);
        textureRegionBounds2.x += delta * transformToScreen(speed / 100 * speedPercentage);
    }

    private void resetBounds() {
        textureRegionBounds1 = textureRegionBounds2;
        textureRegionBounds2 = new Rectangle(Constants.APP_WIDTH + textureRegionBounds1.x, 0, Constants.APP_WIDTH, Constants.APP_HEIGHT);
    }
    protected float transformToScreen(float n) {
        return Constants.WORLD_TO_SCREEN * n;
    }
}