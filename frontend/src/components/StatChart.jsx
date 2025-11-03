import typeColors from "./typeColors"
import {ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, 
    PolarRadiusAxis, Legend} from 'recharts'


export default function StatChart(props) {
    const radarData = [
        { stat: 'HP', value: props.stats[0]},
        { stat: 'ATK', value: props.stats[1]},
        { stat: 'DEF', value: props.stats[2]},
        { stat: 'SP. ATK', value: props.stats[3]},
        { stat: 'SP. DEF', value: props.stats[4]},
        { stat: 'SPEED', value: props.stats[5]}
    ]

    const primaryType = props.types[0]
    const secondaryType = props.types[1]
    
    return (<ResponsiveContainer width = "100%" height = {300}>
                <RadarChart data = {radarData}>
                    <PolarAngleAxis dataKey = "stat" tick={{ fill: "#ffffff", fontSize: 14, fontWeight: "bold" }} />
                    <PolarGrid/>
                    <PolarRadiusAxis tick = {{ fill: "#ffffff", fontSize: 12 }} domain = {[0, 200]}/>
                    <Radar dataKey = "value"
                    fill = {(primaryType === "dark" && secondaryType) ? 
                        typeColors[secondaryType].background : typeColors[primaryType].background}
                    fillOpacity = {0.5}/>
                </RadarChart>
            </ResponsiveContainer>
    )
}