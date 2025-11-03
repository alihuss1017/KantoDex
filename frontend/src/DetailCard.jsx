import { useState, useEffect } from "react"
import StatChart from "./components/StatChart"
import PokeTable from "./components/PokeTable"
import typeColors from "./components/typeColors"

export default function DetailCard(props) {
    const [pokeDetails, setPokeDetails] = useState({})
    const pokeStats = pokeDetails?.stats?.map((stat) => stat.base_stat)
    const pokeTypes = pokeDetails?.types?.map((type) => type.type.name)
    const pokeAbilities = pokeDetails?.abilities?.map((ability) => ability.ability.name)

    const containerStyle = {
                        backgroundColor: typeColors?.[pokeTypes?.[0]]?.color,
                        }
                        
    useEffect(function() {
    fetch(`https://pokeapi.co/api/v2/pokemon/${props.pokemon}`)
    .then(res => res.json())
    .then(data => setPokeDetails(data))
  }, []) 

  return <>
          <div style = {containerStyle} className = "poke-page-container">
            <h1>#{pokeDetails.order} : {props.pokemon}</h1>
            <div className = "sprite-chart-table-container">
              <div>
                <img src = {`https://img.pokemondb.net/sprites/black-white/normal/${props.pokemon.toLowerCase()}.png`}
                  alt={props.pokemon}/>
              </div>
              <div>
                {pokeStats && pokeTypes && pokeAbilities && pokeDetails.height 
                  && pokeDetails.weight && pokeDetails.height && pokeDetails.species &&
                  <PokeTable height = {pokeDetails.height} weight = {pokeDetails.weight}
                  species = {pokeDetails.species} types = {pokeTypes} stats = {pokeStats} 
                  abilities = {pokeAbilities}/>}
              </div>
              <div>
                {pokeStats && pokeTypes && <StatChart types = {pokeTypes} stats = {pokeStats}/>}
              </div>
            </div>
          </div>
            
         </>
}
