package com.useradgents.burgers.test.managers;

import com.useradgents.burgers.test.app.FootApp;
import com.useradgents.burgers.test.mvp.models.Post;
import com.useradgents.burgers.test.mvp.models.Posts;

import javax.inject.Inject;
import javax.inject.Singleton;

@Singleton
public class NewsManager {

    Posts posts;

    @Inject
    public NewsManager() {
        FootApp.getAppComponent().inject(this);
    }

    public Posts getPosts() {
        return posts;
    }

    public void setPosts(Posts posts) {
        this.posts = posts;
    }

    public Post getPostById(String id) {
        for (Post p : posts.getPost()) {
            if (p.getUuid().equals(id)) {
                return p;
            }
        }
        return null;
    }
}