using Microsoft.EntityFrameworkCore;
using ModelLayer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RepositoryLayer
{
    public class Repository
    {
        private readonly DbContextClass _dbContext;
        DbSet<User> users;
        DbSet<Location> locations;
        DbSet<Order> orders;
        DbSet<Product> products;
        DbSet<Inventory> inventories;
        DbSet<Cart> carts;

        // For Testing Populating Database
        //List<Location> storeList = new List<Location>();
        public Repository(DbContextClass dbContextClass) {
            _dbContext = dbContextClass;
            this.users = _dbContext.users;
            this.locations = _dbContext.locations;
            this.orders = _dbContext.orders;
            this.products = _dbContext.products;
            this.inventories = _dbContext.inventories;
            this.carts = _dbContext.carts;
            PopulateDb();
        }


                        //***** USER METHODS *****//
        public User GetUserById(Guid userId)
        {
            User user = users.FirstOrDefault(x => x.UserId == userId);// check if the user is in the Db
            return user;
        }

        /// <summary>
        /// Takes a User and returns the edited version of the User after saving it to the Db.
        /// </summary>
        /// <param name="user"></param>
        /// <returns></returns>
        public User EditUser(User user)
        {
            // search Db for the user
            User user1 = GetUserById(user.UserId);

            // transfer over all the new values
            user1.Fname = user.Fname;
            user1.Lname = user.Lname;
            user1.DefaultStoreId = user.DefaultStoreId;
            user1.Email = user.Email;
            _dbContext.SaveChanges();

            // search the User again to verify that the new User is in the Db
            User user2 = GetUserById(user1.UserId);
            // return the edited User
            return user2;
        }
        public User LoginUser(User user)
        {
            User user1 = users.FirstOrDefault(x => x.Fname == user.Fname && x.Lname == user.Lname);

            if (user1 == null)
            {
                user1 = new User()
                {
                    Fname = user.Fname,
                    Lname = user.Lname,
                    DefaultStoreId = GetDefaultLocationId()
                };
                users.Add(user1);
                _dbContext.SaveChanges();
            }
            return user1;
        }

        public Guid GetDefaultLocationId()
        {
            Location defaultLocation = locations.FirstOrDefault();
            return defaultLocation.LocationId;
        }

        public List<User> GetAllUsers()
        {
            return users.ToList();
        }
        




                        //***** INVENTORY METHODS *****//
        public List<Inventory> GetAllInventories() {
            return inventories.ToList();
        }
        public Inventory GetInventoryById(Guid inventoryId)
        {
            Inventory inventory = new Inventory();
            foreach (Inventory x in inventories)
            {
                if (x.InventoryId == inventoryId)
                {
                    inventory = x;
                }
            }
            return inventory;
        }

        public List<Inventory> GetLocationProducts(Guid locationId)
        {
            List<Inventory> filteredInventories = new List<Inventory>();
            foreach (Inventory x in inventories)
            {
                if(x.LocationId == locationId)
                {
                    filteredInventories.Add(x);
                }
            }
            return filteredInventories;
        }



                        //***** LOCATION METHODS *****//

        public bool DoesLocationExist(Guid locationId)
        {
            if(locations.Any(x => x.LocationId == locationId))
            {
                return true;
            } else
            {
                return false;
            }
        }
        public List<Location> GetAllLocations()
        {
            return locations.ToList();
        }
        public Location GetLocationById(Guid locationtId)
        {
            Location location = new Location();
            foreach (Location x in locations)
            {
                if (x.LocationId == locationtId)
                {
                    location = x;
                }
            }
            return location;
        }



                        //***** PRODUCT METHODS *****//
        public List<Product> GetAllProducts()
        {
            return products.ToList();
        }

        public Product GetProductById(Guid productId)
        {
            Product product = new Product();
            foreach (Product x in products)
            {
                if (x.ProductId == productId)
                {
                    product = x;
                }
            }
            return product;
        }



                        //***** ORDER METHODS *****//
        public List<Order> GetAllOrders()
        {
            return orders.ToList();
        }

        public Order GetOrderById(Guid orderId)
        {
            Order order = new Order();
            foreach (Order x in orders)
            {
                if (x.OrderId == orderId)
                {
                    order = x;
                }
            }
            return order;
        }
        //public Order AddOrder(Order order)
        //{
        //    orders.Add(order);
        //    _dbContext.SaveChanges();
        //    return order;
        //}
        public bool DeleteOrder(Order order)
        {
            orders.Remove(order);
            _dbContext.SaveChanges();
            return true;
        }


        public void DecrementInventory(Guid locationId, Guid productId)
        {
            Inventory inventoryItem = new Inventory();
            foreach(Inventory x in inventories)
            {
                if(x.LocationId == locationId)
                {
                    if(x.ProductId == productId)
                    {
                        x.Quantity--;
                    }
                }
            }
        }




                        //***** CART METHODS *****//
        public List<Cart> GetAllCarts()
        {
            return carts.ToList();
        }

        public Cart GetCartById(Guid cartId)
        {
            Cart cart = new Cart();
            cart.CartId = cartId;
            foreach (Cart x in carts)
            {
                if (x.CartId == cartId)
                {
                    cart = x;
                }
            }
            return cart;
        }

        public List<Cart> GetCartByUser(Guid userId)
        {
            List<Cart> userCarts = new List<Cart>();
            foreach (Cart x in carts)
            {
                if(x.UserId == userId)
                {
                    userCarts.Add(x);
                }
            }
            return userCarts;
        }

        public Cart AddCart(Cart cart)
        {
            carts.Add(cart);
            _dbContext.SaveChanges();
            return cart;
        }




        public bool DoesCartExist(Guid cartId)
        {
            if (carts.Any(x => x.CartId == cartId))
            {
                return true;
            }
            else
            {
                return false;
            }
        }






















        // Populate the Database if Empty
        //
        //
        /// <summary>
        /// Checks to see if Locations Table in Database is Empty
        /// If it is Empty populate with Data
        /// </summary>        
        public void ValidateLocationTable()
        {
            if (locations.Count() == 0)
            {
                locations.Add(new Location("Neverwinter"));
                locations.Add(new Location("Waterdeep"));
                locations.Add(new Location("Baldurs Gate"));
            }
            SaveChanges();
            //foreach (Location store in storeList)
            //{
            //    storeList.Add(store);
            //}
        }
        /// <summary>
        /// Checks to see if Products Table in Database is Empty
        /// If it is Empty populate with Data
        /// </summary>        
        public void ValidateProductTable()
        {
            if (products.Count() == 0)
            {
                products.Add(new Product("Melf's Minute Meteors", 1000, "You create six tiny meteors in your space."));
                products.Add(new Product("Acid Arrow", 500, "A shimmering green arrow streaks toward a target within range and bursts in a spray of acid."));
                products.Add(new Product("Unicorn Arrow", 600, "A translucent unicorn shape appears in midair and speeds toward the target of this spell."));
                products.Add(new Product("Universal Astronomy", 300, "A Book on Universal Astronomy. I'd tell you what''s in it, but no spoilers..."));
                products.Add(new Product("Weapons of the Ether", 300, "A Book on Weapons of the Either. I''d tell you what''s in it, but no spoilers..."));
            }
            SaveChanges();
            //foreach (Product store in products)
            //{
            //    products.Add(store);
            //}
        }

        public void SaveChanges()
        {
            _dbContext.SaveChanges();
        }

        /// <summary>
        /// Calls all of the Methods Which Populate the Database with data if there is not already data in their respective table.
        /// </summary>
        public void PopulateDb()
        {
            ValidateLocationTable();
            ValidateProductTable();
            if (inventories.Count() == 0)
            {
                foreach (Location store in locations.ToList())
                {
                    if (store.InventoryItems.Count() < 1)
                    {
                        foreach (Product product in products.ToList())
                        {
                            AddInventoryItem(store, product, 10);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Adds the selected Inventory Item to the Orders table in Context.
        /// </summary>
        /// <param name="store"></param>
        /// <param name="product"></param>
        /// <param name="amount"></param>
        /// <returns></returns>
        public bool AddInventoryItem(Location store, Product product, int amount)
        {
            Inventory stockItem = new Inventory
            {
                LocationId = store.LocationId,
                ProductId = product.ProductId,
                Quantity = amount
            };
            inventories.Add(stockItem);
            SaveChanges();
            return true;
        }

        /// <summary>
        /// Creates a New Order and Adds it to the Database
        /// </summary>
        /// <param name="store"></param>
        /// <param name="product"></param>
        /// <param name="user"></param>
        /// <param name="cart"></param>
        /// <returns></returns>
        public bool AddOrderItem(Location store, Product product, User user, Cart cart)
        {
            //Guid tempUserId = Guid.Parse("aac98d45-c0df-4e30-ab3f-81beaa2b381d");

            if (!DoesCartExist(cart.CartId))
            {
                cart.StoreId = store.LocationId;
                cart.UserId = user.UserId;
                cart.CartStatus = "Active";
                AddCart(cart);
                //return true;
            }
            Order orderItem = new Order
            {
                LocationId = store.LocationId,
                UserId = user.UserId,
                ProductId = product.ProductId,
                CartId = cart.CartId
            };
            orders.Add(orderItem);
            SaveChanges();
            return true;
        }
    }
}