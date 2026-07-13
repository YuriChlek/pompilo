import { INestApplication } from '@nestjs/common';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';

export const setupSwagger = (app: INestApplication): void => {
    const config = new DocumentBuilder()
        .setTitle('Identity API')
        .setDescription('Identity and registration API for the trading platform.')
        .setVersion('0.1.0')
        .build();

    const swaggerDocument = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('swagger', app, swaggerDocument);
};
