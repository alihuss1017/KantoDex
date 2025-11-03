import pokeDexImg from './assets/pokedex.png'

export default function Header() {
    return <header>
                <img src =  {pokeDexImg}></img>
                <h1> KantoDex: A 1st Generation Pokémon Classifier</h1>
           </header>
}