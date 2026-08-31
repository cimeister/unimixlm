package myJStuff;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionListener;

import javax.swing.JPanel;
import javax.swing.UIManager;
import javax.swing.UIManager.LookAndFeelInfo;
import javax.swing.border.Border;
import javax.swing.border.EmptyBorder;

import net.miginfocom.swing.MigLayout;
import util.ColorUtil;

public abstract class MyPanel{

	protected ActionListener packageListener;
	
	protected JPanel contentPane;
	
	protected JPanel north;
	protected JPanel south;
	protected JPanel west;
	protected JPanel east;
	protected JPanel center;
	
	protected static final Border emptyBorder = new EmptyBorder(1,1,1,1);
	protected static final Border defaultBorder = new EmptyBorder(3,3,3,3);
	
	protected int nationFontSize;
	protected int nationInfoFontSize;
	protected int unitFontSize;
	protected int btnFontSize;
	
	protected final String source = "src/resources/";
	
	protected static final Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
	protected static final int screenWidth = (int) screenSize.getWidth();
	protected static final int screenHeight = (int) screenSize.getHeight();

	public MyPanel(){
		
		setFont();
		setTheme();
		
		contentPane = new JPanel();
		contentPane.setBorder(defaultBorder);
		contentPane.setLayout(new BorderLayout(0, 0));
		
		//These panels are what all JLabels, buttons etc. are added to
		north = new JPanel();
		contentPane.add(north, BorderLayout.NORTH);
		north.setLayout(new MigLayout("insets 0,gapy 0,gapx 0", "[grow,center]", "[]0[]"));
		
		west = new JPanel();
		contentPane.add(west, BorderLayout.WEST);
		west.setLayout(new MigLayout("insets 0,gapy 0","[]",""));
		
		east = new JPanel();
		contentPane.add(east, BorderLayout.EAST);
		east.setLayout(new MigLayout("insets 0,gapy 0","",""));
		
		center = new JPanel();
		contentPane.add(center, BorderLayout.CENTER);
		center.setLayout(new MigLayout("insets 0,gapy 0", "[grow,center]", "[]"));
		
		south = new JPanel();
		contentPane.add(south, BorderLayout.SOUTH);
		south.setLayout(new MigLayout("gapy 1", "[][][grow]", "[]0[]"));
		
		//Once panels are set, background color is then set
		setMainBackground(ColorUtil.backgroundColor);
	}
	
	// Default sizes for certain font templates
	private void setFont(){
		nationFontSize = screenWidth/24;
		nationInfoFontSize = nationFontSize*3/4;
		
		unitFontSize = screenHeight/44;
		btnFontSize = unitFontSize*5/3;
	}
	
	// Ensure the background color for all panels are black
	public void setMainBackground(Color c){
		north.setBackground(c);
		south.setBackground(c);
		east.setBackground(c);
		west.setBackground(c);
		center.setBackground(c);
		contentPane.setBackground(c);
	}
	
	private void setTheme(){
		try {
		    for (LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
		        if ("Nimbus".equals(info.getName())) {
		            UIManager.setLookAndFeel(info.getClassName());
		            break;
		        }
		    }
		} catch (Exception e) {
		    // If Nimbus is not available, you can set the GUI to another look and feel.
		}
	}
	
	public static String convertToMultiline(String orig){
		return "<html>" + orig.replaceAll("\n", "<br>");
	}
	
	public JPanel getContentPane(){
		return contentPane;
	}
	
	public static int getScreenWidth(){
		return screenWidth;
	}
	
	public static int getScreenHeight(){
		return screenHeight;
	}
}