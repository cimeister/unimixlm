package dk.adamino.rehabilitation.GUI.Model;

import dk.adamino.rehabilitation.BLL.FirebaseFacade;
import dk.adamino.rehabilitation.Callbacks.IFirebaseAuthenticationCallback;
import dk.adamino.rehabilitation.Callbacks.IFirestoreClientCallback;
import dk.adamino.rehabilitation.Callbacks.IFirestoreMilestoneCallback;

/**
 * Created by Adamino.
 */
public class FirebaseClientModel {

    private static FirebaseClientModel instance = null;

    private FirebaseFacade mFirebaseFacade;
    private String mCurrentClientUid;

    private FirebaseClientModel() {
        mFirebaseFacade = FirebaseFacade.getInstance();
    }

    public static FirebaseClientModel getInstance() {
        if (instance == null) {
            instance = new FirebaseClientModel();
        }
        return instance;
    }

    /**
     * Set the id of the current logged in client
     *
     * @param currentClientUid
     */
    public void setCurrentClientUid(String currentClientUid) {
        mCurrentClientUid = currentClientUid;
    }

    /**
     * Get current client milestones
     *
     * @param callback
     * @return
     */
    public void getClientMilestones(IFirestoreMilestoneCallback callback) {
        mFirebaseFacade.getFirestoreDAO().getMilestonesByClientId(mCurrentClientUid, callback);
    }

    /**
     * Logout client
     */
    public void logout() {
        mCurrentClientUid = null;
        mFirebaseFacade.getFirebaseAuthenticate().signout();
    }

    /**
     * Login user
     *
     * @param email
     * @param password
     */
    public void loginWithEmailAndPassword(String email, String password, IFirebaseAuthenticationCallback callback) {
        mFirebaseFacade.getFirebaseAuthenticate().signInWithEmailAndPassword(email, password, callback);
    }

    /***
     * Load currently loggedin client
     * @param response
     */
    public void loadLoggedInClientAsync(IFirestoreClientCallback response) {
        mFirebaseFacade.getFirestoreDAO().getClientById(mCurrentClientUid, response);
    }
}