import { Jimp } from "jimp";

async function test() {
    try {
        const img = new Jimp({ width: 100, height: 100 });
        console.log("Success new Jimp without color");
    } catch (e) {
        console.error(e);
    }
}
test();
