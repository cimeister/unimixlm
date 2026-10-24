import { useContext } from "react";
import { ThemeContext } from "../../contexts/ThemeContext";
import "./styles.css";

const white =
  "<KEY>;

const black =
  "<KEY>;

export default function Placeholder() {
  const theme = useContext(ThemeContext);
  return (
    <img
      src={`data:image/png;base64,${
        theme.colorTheme === "dark" ? black : white
      }`}
      alt="placeholder"
      className="placeholder-image"
    />
  );
}