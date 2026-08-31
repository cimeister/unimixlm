const { gql } = require("apollo-server-express");

const typeDefs = gql`
  type User {
    _id: ID
    username: String
    email: String
    password: String
    savedRecipes: [Recipes]!
  }

  # typeDefs for Recipes
  type Recipes {
    _id: ID
    recipeId: String
    recipeName: String
    servings: String
    cookTime: String
    prepTime: String
    specialTools: [String]
    ingredients: [String]
    instructions: [String]
    recipeAuthor: String
    ratings: [String]
  }

  # input data for saved recipes
  input savedRecipesInfo {
    recipeId: String!
    recipeName: String
    servings: String
    cookTime: String
    prepTime: String
    specialTools: [String]
    ingredients: [String]
    instructions: [String]
    recipeAuthor: String
    ratings: [String]
  }

  # auth
  type Auth {
    token: ID!
    user: User
  }

  # Declared queries that can be used, data successfully retrieved
  type Query {
    # tested
    allUsers: [User]
    #Tested
    user(username: String!): User
    #Tested
    allRecipes: [Recipes]
    #tested
    oneRecipe(recipeId: ID!): Recipes
    # thoughts(username: String): [Thought]
    # thought(thoughtId: ID!): Thought
    me: User
  }

  # mutations to alter data
  type Mutation {
    addUser(username: String!, email: String!, password: String!): Auth
    login(email: String!, password: String!): Auth
    saveRecipe(recipeData: savedRecipesInfo!): User
    removeRecipe(recipeId: String!): User
    addRecipe(
      recipeName: String!,
      servings: String,
      prepTime: String,
      cookTime: String,
      specialTools: String,
      ingredients: [String],
      instructions: [String]
    ): Recipes
    # addThought(thoughtText: String!): Thought
    # addComment(thoughtId: ID!, commentText: String!): Thought
    # removeThought(thoughtId: ID!): Thought
    # removeComment(thoughtId: ID!, commentId: ID!): Thought
  }
`;

module.exports = typeDefs;