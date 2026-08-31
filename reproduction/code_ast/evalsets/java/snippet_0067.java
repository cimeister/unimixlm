package ajedrez;

public class Knight {
	public static boolean decideColor=true;
	private Colours colour;
	public Knight() {
		if(decideColor) {
			colour=Colours.WHITE;
		}else {
			colour=Colours.BLACK;
		}
		
	}
}