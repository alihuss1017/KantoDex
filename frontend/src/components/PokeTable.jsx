import typeColors from "./typeColors"

export default function PokeTable(props) {

    let style1 = {
                  backgroundColor: typeColors[props.types[0]].background,
                  color: typeColors[props.types[0]].color,
                }
                
    let style2 = {
                  backgroundColor: props.types[1] ? typeColors[props.types[1]].background : "",
                  color: props.types[1] ? typeColors[props.types[1]].color : "",      
                  marginLeft: "5px"
                }

    

    return <table className = "table-item">
            <tbody>
                <tr>
                    <th scope = "row">Height</th>
                    <td>
                        <span>{parseInt(props.height) * 0.1}m</span>
                    </td>
                </tr>
            </tbody>

            <tbody>
                <tr>
                    <th scope = "row">Weight</th>
                    <td>
                        <span>{parseInt(props.weight) * 0.1}kg</span>
                    </td>
                </tr>
            </tbody>
            
            <tbody>
                <tr>
                    <th scope = "row">Type</th>
                    <td>
                        <span style = {style1}>{props.types[0]}</span>
                        <span style = {props.types[1] ? style2 : null}>{props.types[1]}</span>
                    </td>
                </tr>
            </tbody>
            
            <tbody>
                <tr>
                    <th scope = "row">Abilities</th>
                    <td>{props.abilities.map((ability) => <p>{ability}</p>)}</td>
                </tr>
            </tbody>

            <tbody>
            <tr>
                <th scope = "row">BST</th>
                <td>{(props.stats.reduce((bst, stat) => bst + parseInt(stat)))}</td>
            </tr>
            </tbody>

           </table>
}