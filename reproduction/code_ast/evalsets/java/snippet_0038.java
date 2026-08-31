import java.io.File;

/**
 * ReadDir
 */
public class ReadDir {

    public static void main(String[] args) {
        File file = null;
        String[] paths;
        try {
            file = new File("/home/shahraaz/Documents/JavaCodes/Java-Object_Oriented_Programming/IO_Streams");
            paths = file.list();
            for (String path : paths) {
                System.out.println(path);
            }
        } catch (Exception e) {
            //TODO: handle exception
            e.printStackTrace();
        }
    }
}