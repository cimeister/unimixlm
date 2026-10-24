using System;

/// <summary>
/// DataType representing an bin containing a number of item's.
/// 1.) Initialising create's empty template.
/// 2.) Use Load to fill in variable's.
/// </summary>
public class bin
{
    string[,,] stringBin = new string[2, 4, 2];
    ioINI iniDetails = new ioINI();

    //Bin Information
    public string _Path;
    public string _File;
    int _Store;
    int _Level;
    int _Shelf;
    int _Bin;
    int _Quantity;
    int _Price;
    //Article Information
    int _Storage;
    int _Section;
    int _Article;
    article articleSelected;
    
    /// <summary>
    /// Retrieve Value from given Category && Option.
    /// </summary>
    /// <param name="Category">Category as string.</param>
    /// <param name="Option">Option as string.</param>
    /// <returns>Single string with Value.</returns>
    string Retrieve(string Category, string Option)
    {
        for (int intC = 0; intC <= stringBin.GetUpperBound(0); intC++) {
            if (stringBin[intC, 0, 0] == Category) {
                for (int intO = 0; intO <= stringBin.GetUpperBound(1); intO++) {
                    if (stringBin[intC, intO, 0] == Option) {
                        return stringBin[intC, intO, 1];
                    }
                }
            }
        }
        return null;
    }

    /// <summary>
    /// Change the Value of a Option in a given Category.
    /// </summary>
    /// <param name="Category">Category as string.</param>
    /// <param name="Option">Option as string.</param>
    /// <param name="Value">Value as string.</param>
    void Change(string Category, string Option, string Value)
    {
        for (int intC = 0; intC <= stringBin.GetUpperBound(0); intC++) {
            if (stringBin[intC, 0, 0] == Category) {
                for (int intO = 0; intO <= stringBin.GetUpperBound(1); intO++) {
                    if (stringBin[intC, intO, 0] == Option) {
                        stringBin[intC, intO, 1] = Value;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Loads the bin through Store, Level, Shelf and Bin.
    /// </summary>
    /// <param name="Store">Store of Bin.</param>
    /// <param name="Level">Level of Bin.</param>
    /// <param name="Shelf">Shelf of Bin.</param>
    /// <param name="Bin">Designated Bin.</param>
    public void Load(int Store, int Level, int Shelf, int Bin)
    {
        _Store = Store;
        _Level = Level;
        _Shelf = Shelf;
        _Bin = Bin;

        // Check if bin exists, yes=load no=new.
        if (System.IO.File.Exists(System.IO.Directory.GetCurrentDirectory() + (char)92 + "save" + (char)92 + gamecache.currentPlayer.ProfileID + (char)92 + "store" + (char)92 + Store + (char)92 + Level + (char)92 + Shelf + (char)92 + Bin + (char)92 + "details.ini")) {
            //Bin Information
            _Store = Store;
            _Level = Level;
            _Shelf = Shelf;
            _Bin = Bin;
            _Path = "save" + (char)92 + gamecache.currentPlayer.ProfileID + (char)92 + "store" + (char)92 + Store + (char)92 + Level + (char)92 + Shelf + (Char)92 + Bin;
            _File = "details.ini";
            stringBin = iniDetails.Load(_Path, _File); //Load bin ini
            _Quantity = Convert.ToInt32(Retrieve("Bin Information", "Quantity of article"));
            _Price = Convert.ToInt32(Retrieve("Bin Information", "Price of article"));
            //Article Information
            _Storage = Convert.ToInt32(Retrieve("Article Information", "Storage of article"));
            _Section = Convert.ToInt32(Retrieve("Article Information", "Section of article"));
            _Article = Convert.ToInt32(Retrieve("Article Information", "Designated article"));
            articleSelected = new article();
            articleSelected.Load(_Storage, _Section, _Article);
            
        } else {
            //Bin Information
            stringBin[0, 0, 0] = "Bin Information";
            stringBin[0, 1, 0] = "Quantity of article";
            stringBin[0, 1, 1] = "0";
            _Quantity = 0;
            stringBin[0, 2, 0] = "Price of article";
            stringBin[0, 2, 1] = "0";
            _Price = 0;
            //Article Information
            stringBin[1, 0, 0] = "Article Information";
            stringBin[1, 1, 0] = "Storage of article";
            stringBin[1, 1, 1] = "0";
            _Storage = 0;
            stringBin[1, 2, 0] = "Section of article";
            stringBin[1, 2, 1] = "0";
            _Section = 0;
            stringBin[1, 3, 0] = "Designated article";
            stringBin[1, 3, 1] = "0";
            _Article = 0;
            binSave();
        }
    }
    
    void binSave()
    {
        iniDetails.Save("save" + (char)92 + gamecache.currentPlayer.ProfileID + (char)92 + "store" + (char)92 + _Store + (char)92 + _Level + (char)92 + _Shelf + (char)92 + _Bin, "details.ini", stringBin);
    }

    /// <summary>
    /// Get/Sets the Quantity of the Article in the bin-slot WITH Change() and binSave().
    /// </summary>
    public int Quantity {
        get {
            return _Quantity;
        }
        set {
            _Quantity = value;
            Change("Bin Information", "Quantity of article", value.ToString());
            binSave();
        }
    }
    
    /// <summary>
    /// Get/Sets the Price of the article in the bin-slot WITH Change() and binSave().
    /// </summary>
    public int Price {
        get {
            return _Price;
        }
        set {
            _Price = value;
            Change("Bin Information", "Price of article", value.ToString());
            binSave();
        }
    }

    /// <summary>
    /// Get/Sets the Storage of the article in the bin-slot WITH Change() and WITHOUT binSave() & articleSelected.Load().
    /// When setting the designated article it will do binSave() & articleSelected.Load().
    /// </summary>
    public int Storage {
        get {
            return _Storage;
        }
        set {
            _Storage = value;
            Change("Article Information", "Storage of article", value.ToString());
        }
    }

    /// <summary>
    /// Get/Sets the Section of the article in the bin-slot WITH Change() and WITHOUT binSave() & articleSelected.Load().
    /// When setting the designated article it will do binSave() & articleSelected.Load().
    /// </summary>
    public int Section {
        get {
            return _Section;
        }
        set {
            _Section = value;
            Change("Article Information", "Section of article", value.ToString());
        }
    }

    /// <summary>
    /// Get/Sets the designated article in the bin-slot WITH Change(), binSave() & articleSelected.Load().
    /// </summary>
    public int Article {
        get {
            return _Article;
        }
        set {
            _Article = value;
            Change("Article Information", "Designated article", value.ToString());
            binSave();
            articleSelected.Load(_Storage, _Section, _Article);
        }
    }

    /// <summary>
    /// Get the article datatype. For setting use Storage(), Section() & Article().
    /// </summary>
    public article Article_Data {
        get {
            return articleSelected;
        }
    }
}