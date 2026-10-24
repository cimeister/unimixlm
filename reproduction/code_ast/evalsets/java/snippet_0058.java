//import org.junit.jupiter.api.Test;
//import org.springframework.boot.test.context.SpringBootTest;
//
//@SpringBootTest
//class ElectricPowerApplicationTests {
//
//    @Test
//    void contextLoads() {
//    }
//
//    public static final int FLAG = -32523523;
//
//    /**
//     * byte数组转hex
//     * @param bytes
//     * @return
//     */
//    public static String byteToHex(byte[] bytes){
//        String strHex = "";
//        StringBuilder sb = new StringBuilder("");
//        for (int n = 0; n < bytes.length; n++) {
//            strHex = Integer.toHexString(bytes[n] & 0xFF);
//            sb.append((strHex.length() == 1) ? "0" + strHex : strHex); // 每个字节由两个字符表示，位数不够，高位补0
//        }
//        return sb.toString().trim();
//    }
//
//
//    /**
//     * hex转byte数组
//     * @param hex
//     * @return
//     */
//    public static byte[] hexToByte(String hex){
//        int m = 0, n = 0;
//        int byteLen = hex.length() / 2; // 每两个字符描述一个字节
//        byte[] ret = new byte[byteLen];
//        for (int i = 0; i < byteLen; i++) {
//            m = i * 2 + 1;
//            n = m + 1;
//            System.out.println("0x" + hex.substring(i * 2, m) + hex.substring(m, n));
//            int intVal = Integer.decode("0x" + hex.substring(i * 2, m) + hex.substring(m, n));
//            ret[i] = Byte.valueOf((byte)intVal);
//        }
//        return ret;
//    }
//
//
//
//    //fe0fbafd
//    //0xfe
//    //0x0f
//    //0xba
//    //0xfd
//    public static void main(String[] args) {
//        //integer -> hex str -> byte arrs
//        String s = Integer.toHexString(FLAG);
//        System.out.println(s);
//        System.out.println(hexToByte(s).toString());
//        System.out.println("===========================");
//        // byte arrs -> hex str -> integer
//        byte[] bytes=new byte[]{(byte) 0xfe,(byte)0x0f, (byte) 0xba, (byte) 0xfd};
//        String s1 = byteToHex(bytes);
//        System.out.println(s1);
//        System.out.println(Integer.parseInt("FE0FBAFD", 10));
//
//
//    }
//
//}