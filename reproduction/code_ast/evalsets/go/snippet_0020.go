// Copyright 2016 <NAME>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Package block TODO doc

package block

import (
	"fmt"
	"runtime"

	"github.com/go-gl/mathgl/mgl32"
	"github.com/hurricanerix/shade/entity"
	"github.com/hurricanerix/shade/fonts"
	"github.com/hurricanerix/shade/shapes"
	"github.com/hurricanerix/shade/sprite"
)

func init() {
	// GLFW event handling must run on the main OS thread
	runtime.LockOSThread()
}

// Player TODO doc
type Block struct {
	pos    mgl32.Vec3
	Sprite *sprite.Context
	Font   fonts.Context
	Style  float32
	Shape  *shapes.Shape
}

// New TODO doc
func New(style, x, y float32, s *sprite.Context, f fonts.Context) Block {
	b := Block{
		pos:    mgl32.Vec3{x, y, 1},
		Sprite: s,
		Font:   f,
		Style:  style,
		Shape:  shapes.NewRect(0, float32(s.Width), 0, float32(s.Height)),
	}
	return b
}

func (b Block) Bounds() shapes.Shape {
	return *b.Shape
}

func (b Block) Pos() mgl32.Vec3 {
	return b.pos
}

// Bind TODO doc
func (b *Block) Bind(program uint32) error {
	return b.Sprite.Bind(program)
}

// Update TODO doc
func (b *Block) Update(dt float32, g []entity.Collider) {
	// Blocks don't do anything
}

// Draw TODO doc
func (b Block) Draw() {
	efx := sprite.Effects{
		Scale: mgl32.Vec3{2.0, 2.0, 1.0},
	}

	msg := fmt.Sprintf("Pos: (%.0f,%.0f)\n", b.pos[0], b.pos[1])
	msg += fmt.Sprintf("Data: [\n")
	msg += fmt.Sprintf("  Left: %.0f\n", b.Shape.Data[0])
	msg += fmt.Sprintf("  Right: %.0f\n", b.Shape.Data[1])
	msg += fmt.Sprintf("  Top: %.0f\n", b.Shape.Data[2])
	msg += fmt.Sprintf("  Bottom: %.0f\n", b.Shape.Data[3])
	msg += fmt.Sprintf("]\n")
	_, h := b.Font.SizeText(&efx, msg)
	//b.Font.DrawText(mgl32.Vec3{0, b.pos[1] + 32 - 16, 0}, &efx, msg)
	b.Font.DrawText(mgl32.Vec3{b.pos[0], b.pos[1] + h + 7, 0}, &efx, msg)

	b.Sprite.DrawFrame(mgl32.Vec2{b.Style, 0}, mgl32.Vec3{b.pos[0], b.pos[1], 1.0}, nil)
}