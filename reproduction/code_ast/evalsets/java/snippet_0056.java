package kfs.api;

/**
 *
 * @author pavedrim
 */
public interface ImportTable {
    public boolean addDataPart(int dayNo, String filename);
    public void deletePartTable();
    public void rebuildIndexies();
    
}