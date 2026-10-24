using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;

namespace Sprint2
{
    public class ProjectilePhysicalProperty
    {
        private IProjectile projectile;
        public Vector2 Velocity { get; set; }
        public Vector2 Accerlation1 { get; set; }
        public Vector2 Accerlation2 { get; set; }
        public Direction direction { get; set; }
        public Boolean IsUsing { get; set; }
        private readonly int zero = 0;
        private readonly int hundred = 50;
        public ProjectilePhysicalProperty(IProjectile Projectile)
        {
            this.projectile = Projectile;
            Velocity = new Vector2(zero, zero);
            Accerlation1 = new Vector2(zero, hundred);
            Accerlation2 = new Vector2(hundred, zero);
        }

        public void Update(GameTime gametime)
        {
            float deltaTime = (float)gametime.ElapsedGameTime.TotalSeconds;
            
            

                if (direction == Direction.DOWN)
                {
                    Velocity = new Vector2(zero, 100);
                }
                else if (direction == Direction.UP)
                {
                    Velocity = new Vector2(zero, -100);
                }
                else if (direction == Direction.RIGHT)
                {
                    Velocity = new Vector2(100, zero);
                }
                else
                {
                    Velocity = new Vector2(-100, zero);
                }
            projectile.Position += Velocity * deltaTime;

        }
            
        }

 }